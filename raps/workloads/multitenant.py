import random
from raps.job import Job, job_dict
from .constants import ACCT_NAMES, MAX_PRIORITY

class MultitenantWorkload:

    def multitenant(self, **kwargs):
        """
        Generate deterministic jobs to validate multitenant scheduling & power.

        usage example:

            python main.py run-multi-part -x mit_supercloud -w multitenant

        Parameters
        ----------
        mode : str
            One of:
              - 'ONE_JOB_PER_NODE_ALL_CORES'
              - 'TWO_JOBS_PER_NODE_SPLIT'
              - 'STAGGERED_JOBS_PER_NODE'
        wall_time : int
            Duration (seconds) of each job (default: 3600)
        trace_quanta : int
            Sampling interval for traces; defaults to config['TRACE_QUANTA']

        Returns
        -------
        list[dict]
            List of job_dict entries.
        """
        mode = kwargs.get('mode', 'TWO_JOBS_PER_NODE_SPLIT')
        wall_time = kwargs.get('wall_time', 3600)

        jobs = []

        for partition in self.partitions:
            cfg = self.config_map[partition]
            trace_quanta = kwargs.get('trace_quanta', cfg['TRACE_QUANTA'])

            cores_per_cpu = cfg.get('CORES_PER_CPU', 1)
            cpus_per_node = cfg.get('CPUS_PER_NODE', 1)
            cores_per_node = cores_per_cpu * cpus_per_node
            gpus_per_node = cfg.get('GPUS_PER_NODE', 0)

            n_nodes = cfg['AVAILABLE_NODES']

            def make_trace(cpu_util, gpu_util):
                return self.compute_traces(cpu_util, gpu_util, wall_time, trace_quanta)

            job_id_ctr = 0

            if mode == 'ONE_JOB_PER_NODE_ALL_CORES':
                # Each node runs one job that consumes all cores/GPUs
                for nid in range(n_nodes):
                    cpu_trace, gpu_trace = make_trace(cores_per_node, gpus_per_node)
                    jobs.append(Job(job_dict(
                        nodes_required=1,
                        cpu_cores_required=cores_per_node,
                        gpu_units_required=gpus_per_node,
                        name=f"MT_full_node_{partition}_{nid}",
                        account=random.choice(ACCT_NAMES),
                        cpu_trace=cpu_trace,
                        gpu_trace=gpu_trace,
                        ntx_trace=[], nrx_trace=[],
                        end_state='COMPLETED',
                        id=job_id_ctr,
                        priority=random.randint(0, MAX_PRIORITY),
                        partition=partition,
                        submit_time=0,
                        time_limit=wall_time,
                        start_time=0,
                        end_time=wall_time,
                        expected_run_time=wall_time,
                        trace_time=wall_time,
                        trace_start_time=0,
                        trace_end_time=wall_time,
                        trace_quanta=cfg['TRACE_QUANTA']
                    )))
                    job_id_ctr += 1

            elif mode == 'TWO_JOBS_PER_NODE_SPLIT':
                # Two jobs per node: split CPU/GPU roughly in half
                for nid in range(n_nodes):
                    cpu_a = cores_per_node // 2
                    cpu_b = cores_per_node - cpu_a
                    gpu_a = gpus_per_node // 2
                    gpu_b = gpus_per_node - gpu_a

                    for idx, (c_req, g_req, tag) in enumerate([(cpu_a, gpu_a, 'A'),
                                                               (cpu_b, gpu_b, 'B')]):
                        cpu_trace, gpu_trace = make_trace(c_req, g_req)
                        jobs.append(Job(job_dict(
                            nodes_required=1,  # still one node; multitenant RM packs cores
                            cpu_cores_required=c_req,
                            gpu_units_required=g_req,
                            name=f"MT_split_node_{partition}_{nid}_{tag}",
                            account=random.choice(ACCT_NAMES),
                            cpu_trace=cpu_trace,
                            gpu_trace=gpu_trace,
                            ntx_trace=[], nrx_trace=[],
                            end_state='COMPLETED',
                            id=job_id_ctr,
                            priority=random.randint(0, MAX_PRIORITY),
                            partition=partition,
                            submit_time=0,
                            time_limit=wall_time,
                            start_time=0,
                            end_time=wall_time,
                            expected_run_time=wall_time,
                            trace_time=wall_time,
                            trace_start_time=0,
                            trace_end_time=wall_time,
                            trace_quanta=cfg['TRACE_QUANTA']
                        )))
                        job_id_ctr += 1

            elif mode == 'STAGGERED_JOBS_PER_NODE':
                # Three jobs per node, staggered starts: 0, wall_time/3, 2*wall_time/3
                offsets = [0, wall_time // 3, 2 * wall_time // 3]
                cpu_each = cores_per_node // 3 or 1
                gpu_each = max(1, gpus_per_node // 3) if gpus_per_node else 0

                for nid in range(n_nodes):
                    for k, offset in enumerate(offsets):
                        cpu_trace, gpu_trace = make_trace(cpu_each, gpu_each)
                        jobs.append(Job(job_dict(
                            nodes_required=1,
                            cpu_cores_required=cpu_each,
                            gpu_units_required=gpu_each,
                            name=f"MT_stagger_node_{partition}_{nid}_{k}",
                            account=random.choice(ACCT_NAMES),
                            cpu_trace=cpu_trace,
                            gpu_trace=gpu_trace,
                            ntx_trace=[], nrx_trace=[],
                            end_state='COMPLETED',
                            id=job_id_ctr,
                            priority=random.randint(0, MAX_PRIORITY),
                            partition=partition,
                            submit_time=offset,
                            time_limit=wall_time,
                            start_time=offset,
                            end_time=offset + wall_time,
                            expected_run_time=wall_time,
                            trace_time=wall_time,
                            trace_start_time=0,
                            trace_end_time=wall_time,
                            trace_quanta=cfg['TRACE_QUANTA']
                        )))
                        job_id_ctr += 1
            else:
                raise ValueError(f"Unknown multitenant mode: {mode}")

        return jobs
