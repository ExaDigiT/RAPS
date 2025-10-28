"""
Hao Lu’s analytical HPL model adapter for ExaDigiT.

Usage:
    python main.py run -w hpl -d
or:
    python raps/workloads/hpl.py
"""

from raps.job import Job, job_dict
import numpy as np
import math


class HPL:
    """Analytical HPL workload generator for ExaDigiT."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Public entry
    # -------------------------------------------------------------------------
    def hpl(self, **kwargs):
        jobs = []

        # You can add more scenarios; comment out big ones while testing.
        hpl_tests = [
            # Smaller grid (quick sanity check)
            {"M": 2_097_152, "b": 576, "P": 16, "Q": 32, "Rtype": "1-ring", "f": 0.6},
            # Frontier-scale shape (comment in when ready)
            {"M": 8_900_000, "b": 576, "P": 192, "Q": 384, "Rtype": "1-ring", "f": 0.6},
        ]

        for test in hpl_tests:
            for partition in self.partitions:
                cfg = self.config_map[partition]
                trace_quanta = cfg["TRACE_QUANTA"]

                # Per-iteration timings (already concurrency-aware)
                iterations = self._run_hpl_model(**test)

                # Convert iteration timings to sampled traces on TRACE_QUANTA grid
                gpu_trace, cpu_trace = self._emit_traces_from_iters(
                    iterations, trace_quanta, cfg
                )
                total_time = len(gpu_trace) * trace_quanta

                # Node count: ranks / (GPUs_per_node * GCDs_per_GPU)
                gpus = cfg["GPUS_PER_NODE"]
                gcds = cfg.get("GCDS_PER_GPU", 2)  # Frontier MI250X default: 2
                ranks = test["P"] * test["Q"]
                nodes_required = max(1, ranks // (gpus * gcds))

                job_info = job_dict(
                    nodes_required=nodes_required,
                    scheduled_nodes=[],
                    name=f"HPL_{test['M']}x{test['M']}_P{test['P']}Q{test['Q']}",
                    account="benchmark",
                    cpu_trace=cpu_trace,
                    gpu_trace=gpu_trace,
                    ntx_trace=[],
                    nrx_trace=[],
                    id=None,
                    end_state="COMPLETED",
                    priority=100,
                    partition=partition,
                    time_limit=total_time,
                    start_time=0,
                    end_time=total_time,
                    expected_run_time=total_time,
                    trace_quanta=trace_quanta,
                    trace_time=total_time,
                    trace_start_time=0,
                    trace_end_time=total_time,
                )
                jobs.append(Job(job_info))

        return jobs

    # -------------------------------------------------------------------------
    # Analytical per-iteration model (concurrency-aware)
    # -------------------------------------------------------------------------
    def _run_hpl_model(self, M, b, P, Q, Rtype="1-ring", f=0.6):
        """
        Returns a list of dicts, one per iteration:
        {
            "T_iter": <iteration wall time (s)>,
            "gpu_active": <seconds in iteration attributable to GPU UPDATE>,
            "cpu_active": <seconds in iteration attributable to CPU PDFACT>,
            "net_active": <seconds in iteration attributable to collectives>,
        }

        Concurrency-aware scaling:
          - UPDATE (DGEMM) work is distributed over the full P*Q ranks  → divide by (P*Q)
          - PDFACT/LBCAST/RS* progress along process columns (Q)         → divide by Q
        This makes the per-iteration times reflect global wall-time.
        """
        # Effective per-rank throughputs/bandwidths (empirical constants)
        CAllgather = 6.3e9     # bytes/s
        C1ring     = 7.0e9     # bytes/s
        Creduce    = 46e6      # bytes/s
        Fcpublas   = 240e9     # FLOP/s
        Fgemm      = 24e12     # FLOP/s

        Ml = M / P
        Nl = M / Q
        nb = int(M / b)
        iterations = []

        for i in range(nb):
            Ml_i = Ml - (i * b / P)
            if Ml_i <= 0:
                break

            # Local column partition sizes (A = [A1 | A2]), f is the split ratio
            Nl1_i = max((1.0 - f) * Nl - (i * b / Q), 0.0)
            Nl2_i = (f * Nl) if (i * b) < (f * Nl) else max(Nl - (i * b / Q), 0.0)

            # Component times (per-rank formulations)
            # NOTE: units already account for bytes vs. elements (coeffs 16, 2/3, etc.)
            TPDFACT_rank = (b**2) / Creduce + (2.0 / 3.0) * (b**2) * Ml_i / Fcpublas
            TLBCAST_rank = 16.0 * b * Ml_i / C1ring
            TUPD1_rank   = 2.0 * b * Ml_i * Nl1_i / Fgemm
            TUPD2_rank   = 2.0 * b * Ml_i * Nl2_i / Fgemm
            TRS1_rank    = 16.0 * b * Nl1_i / CAllgather
            TRS2_rank    = 16.0 * b * Nl2_i / CAllgather

            # Concurrency: convert rank-local times to global wall-time contributions
            # (coarse but effective partitioning of the communicators)
            TPDFACT = TPDFACT_rank #/ Q
            TLBCAST = TLBCAST_rank #/ Q
            TRS1    = TRS1_rank #/ Q
            TRS2    = TRS2_rank #/ Q
            TUPD1   = TUPD1_rank #/ (P * Q)
            TUPD2   = TUPD2_rank #/ (P * Q)

            # Two pipeline stages per iteration (HPL)
            stage1 = max(TPDFACT + TLBCAST + TRS1, TUPD2)
            stage2 = max(TRS2, TUPD1)
            T_iter = stage1 + stage2

            # Attribute activity (for utilization duty fractions)
            gpu_active = max(TUPD1, TUPD2)
            cpu_active = TPDFACT
            net_active = TLBCAST + TRS1 + TRS2

            iterations.append(
                dict(
                    T_iter=T_iter,
                    gpu_active=gpu_active,
                    cpu_active=cpu_active,
                    net_active=net_active,
                )
            )

        return iterations

    def _emit_traces_from_iters(self, iterations, trace_quanta, cfg):
        gpn = cfg["GPUS_PER_NODE"]
        gpu_trace, cpu_trace = [], []
        acc_time = 0.0
        acc_gpu = 0.0
        acc_cpu = 0.0

        for it in iterations:
            T = it["T_iter"]
            if T <= 0: 
                continue

            total_act = it["gpu_active"] + it["cpu_active"] + it["net_active"]
            compute_ratio = it["gpu_active"] / total_act if total_act > 0 else 0.0
            cpu_ratio = it["cpu_active"] / total_act if total_act > 0 else 0.0
            fg = 0.8 + 0.2 * compute_ratio
            fc = 0.6 + 0.3 * cpu_ratio

            acc_time += T
            acc_gpu += gpn * fg * T
            acc_cpu += fc * T

            # emit one sample each time we accumulate ≥ trace_quanta
            while acc_time >= trace_quanta:
                gpu_trace.append(acc_gpu / acc_time)
                cpu_trace.append(acc_cpu / acc_time)
                acc_time -= trace_quanta
                acc_gpu = acc_cpu = 0.0

        # flush remainder
        if acc_time > 0:
            gpu_trace.append(acc_gpu / acc_time)
            cpu_trace.append(acc_cpu / acc_time)

        return np.array(gpu_trace), np.array(cpu_trace)

# -----------------------------------------------------------------------------
# Stand-alone test
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    class DummyHPL(HPL):
        def __init__(self):
            self.partitions = ["gpu"]
            self.config_map = {
                "gpu": {
                    "TRACE_QUANTA": 15.0,   # seconds/sample
                    "GPUS_PER_NODE": 4,     # Frontier physical GPUs/node
                    "GCDS_PER_GPU": 2,      # MI250X logical ranks/GPU
                    "CPUS_PER_NODE": 64,
                }
            }

    hpl = DummyHPL()
    jobs = hpl.hpl()

    print(f"Generated {len(jobs)} HPL job(s)\n")
    for i, job in enumerate(jobs):
        print(f"--- Job {i} ---")
        print(f"Name: {job.name}")
        print(f"Nodes required: {job.nodes_required}")
        print(f"Wall time: {job.trace_time:.1f}s")
        print(f"Trace samples: {len(job.gpu_trace)}")
        print(f"Avg GPU util: {np.mean(job.gpu_trace):.2f} (0..{hpl.config_map['gpu']['GPUS_PER_NODE']})")
        print(f"Avg CPU util: {np.mean(job.cpu_trace):.2f} (0..1)")
        # Peek at starts/ends
        print("GPU head:", np.round(job.gpu_trace[:8], 3))
        print("GPU tail:", np.round(job.gpu_trace[-8:], 3))
        print("CPU head:", np.round(job.cpu_trace[:8], 3))
        print("CPU tail:", np.round(job.cpu_trace[-8:], 3))
        print()
