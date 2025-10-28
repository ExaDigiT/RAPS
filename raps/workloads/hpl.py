"""
Hao Lu's analytical HPL model. Ref:

    Lu et al., "Insights from Optimizing HPL Performance on Exascale Systems: 
    A Comparative Analysis of Panel Factorization", in SC'25 Proceedings.

Test using:

    python main.py run -w hpl -d

or:
    python raps/workloads/hpl.py    

"""
from raps.job import Job, job_dict
import numpy as np
import math, random, json


class HPL:
    """Analytical HPL workload generator for ExaDigiT"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def hpl(self, **kwargs):
        jobs = []
        # Example: parameter sweep across node counts or block sizes
        hpl_tests = [
            {"M": 1482910, "b": 576, "P": 16, "Q": 32, "Rtype": "1-ring"},
            {"M": 2965820, "b": 576, "P": 32, "Q": 32, "Rtype": "1-ring"},
            {"M": 16777216, "b": 576, "P": 192, "Q": 192, "Rtype": "1-ring"},
        ]

        #GCDS_PER_GPU = 2

        for test in hpl_tests:
            for partition in self.partitions:
                cfg = self.config_map[partition]
                trace_quanta = cfg["TRACE_QUANTA"]

                # --- Analytical model evaluation ---
                results = self._run_hpl_model(**test)

                total_time = results["T_total"]
                gpu_util = results["gpu_util"]
                cpu_util = results["cpu_util"]

                num_samples = math.ceil(total_time / trace_quanta) + 1
                gpu_trace = np.full(num_samples, gpu_util)
                cpu_trace = np.full(num_samples, cpu_util)

                job_info = job_dict(
                    #nodes_required=test["P"] * test["Q"] // (cfg["GPUS_PER_NODE"] * GCDS_PER_GPU),
                    nodes_required=test["P"] * test["Q"] // cfg["GPUS_PER_NODE"],
                    scheduled_nodes=[],
                    name=f"HPL_{test['M']}x{test['M']}",
                    account="benchmark",
                    cpu_trace=cpu_trace,
                    gpu_trace=gpu_trace,
                    ntx_trace=[], nrx_trace=[],
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

    def _run_hpl_model(self, M, b, P, Q, Rtype="1-ring", f=0.6):
        # constants (Table II + Fig 2b)
        CAllgather = 6.3e9
        C1ring = 7e9
        Creduce = 46e6
        Fcpublas = 240e9
        Fgemm = 24e12

        Ml = M / P
        Nl = M / Q
        nb = int(M / b)
        total_T = 0.0

        print("*** nb:", nb)
        for i in range(nb):
            Ml_i = Ml - (i * b / P)
            Nl1_i = max((1 - f) * Nl - i * b / Q, 0)
            Nl2_i = f * Nl if i * b < f * Nl else Nl - i * b / Q

            TPDFACT = b ** 2 / Creduce + (2 / 3) * b ** 2 * Ml_i / Fcpublas
            TLBCAST = 16 * b * Ml_i / C1ring
            TUPD1 = 2 * b * Ml_i * Nl1_i / Fgemm
            TUPD2 = 2 * b * Ml_i * Nl2_i / Fgemm
            TRS1 = 16 * b * Nl1_i / CAllgather
            TRS2 = 16 * b * Nl2_i / CAllgather

            total_T += max(TPDFACT + TLBCAST + TRS1, TUPD2) + max(TRS2, TUPD1)

        # derive synthetic utilization
        gpu_util = min(1.0, (Fgemm / 25e12))      # normalized ratio
        cpu_util = min(1.0, (Fcpublas / 250e9))

        return {"T_total": total_T, "gpu_util": gpu_util, "cpu_util": cpu_util}


if __name__ == "__main__":
    import json
    import numpy as np

    # Mock minimal configuration values to mimic ExaDigiT runtime
    class DummyHPL(HPL):
        def __init__(self):
            # Provide fake partitions and system config
            self.partitions = ["gpu"]
            self.config_map = {
                "gpu": {
                    "TRACE_QUANTA": 15.0,      # seconds per trace tick
                    "GPUS_PER_NODE": 4,
                    "CPUS_PER_NODE": 64,
                }
            }

    # Instantiate dummy workload
    workload = DummyHPL()

    # Run synthetic job generation
    jobs = workload.hpl()

    print(f"Generated {len(jobs)} HPL jobs:\n")
    for i, job in enumerate(jobs):
        print(i, job)
        print(f"--- Job {i} ---")
        print(f"Name: {job.name}")
        print(f"Nodes required: {job.nodes_required}")
        print(f"Wall time: {job.trace_time:.2f} s")
        print(f"CPU trace length: {len(job.cpu_trace)}")
        print(f"GPU trace length: {len(job.gpu_trace)}")
        print(f"Avg CPU util: {np.mean(job.cpu_trace):.3f}")
        print(f"Avg GPU util: {np.mean(job.gpu_trace):.3f}")
        print(f"Expected run time: {job.expected_run_time:.2f}")
        print()
