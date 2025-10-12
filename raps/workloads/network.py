
from raps.job import Job, job_dict

class NetworkTestWorkload:
    def network_test(self, **kwargs):
        """
        A synthetic workload to test network congestion.
        """
        config = kwargs.get('config', {})
        # High network traffic to trigger congestion
        # These values are per-node, and the network simulation sums them up
        # so we need to make them high enough to exceed the total network bandwidth
        net_tx = 1e12  # bytes
        net_rx = 1e12  # bytes

        job_info = job_dict(
            nodes_required=2,
            name="network-test-job",
            account="test",
            cpu_trace=[1],
            gpu_trace=[1],
            ntx_trace=[net_tx],
            nrx_trace=[net_rx],
            end_state='COMPLETED',
            id=1,
            priority=100,
            partition='partition',
            submit_time=0,
            time_limit=3600,
            start_time=0,
            end_time=3600,
            expected_run_time=3600,
            trace_quanta=20,
        )
        job = Job(job_info)
        return [job]
