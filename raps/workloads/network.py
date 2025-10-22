
from raps.job import Job, job_dict


class NetworkTestWorkload:
    def network_test(self, **kwargs):
        """
        Synthetic workload to test network congestion.
        Generates several jobs with varying sizes and bandwidths,
        including overlapping node assignments to induce interference.
        """
        jobs = []
        trace_len = 180  # 15 minutes with 20s quanta

        # --------------------------------------------------------
        # Hard-coded configuration
        # --------------------------------------------------------
        # Define per-job properties
        bw = 1e10
        job_configs = [
            # (job_id, node_list, bandwidth_bytes_per_tick)
            (1, [0, 1], bw),      # 2-node job
#            (2, [1, 2], bw),      # Job 2 overlaps node 1 (causes congestion)
            (2, [128, 129], bw),    # Job 2 on a distant rack (no shared link)
            (3, [256], bw),       # isolated single-node job
            (4, [512, 513, 514], 5e11),  # multi-node but separate
            (5, [1020], bw),      # distant single-node job
        ]

        runtime = 900      # seconds
        time_limit = 1800  # seconds
        trace_quanta = 20  # seconds

        # --------------------------------------------------------
        # Job creation loop
        # --------------------------------------------------------
        for job_id, node_list, bw in job_configs:
            job_info = job_dict(
                id=job_id,
                name=f"net_job_{job_id}",
                account="test",
                nodes_required=len(node_list),
                scheduled_nodes=node_list,
                cpu_trace=[1] * trace_len,
                gpu_trace=[1] * trace_len,
                ntx_trace=[bw] * trace_len,
                nrx_trace=[bw] * trace_len,
                submit_time=0,
                start_time=0,
                expected_run_time=runtime,
                time_limit=time_limit,
                end_state="COMPLETED",
                trace_quanta=trace_quanta,
            )
            jobs.append(Job(job_info))
            print(f"[DEBUG] Created net_job_{job_id} nodes={node_list} bw={bw:.2e}")

        print("\n[DEBUG] Requested node assignments:")
        for job in jobs:
            print(f"  Job {job.id}: nodes_required={job.nodes_required}, scheduled_nodes={job.scheduled_nodes}")

        return jobs
