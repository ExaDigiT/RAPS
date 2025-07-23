"""
Module for generating workload traces and jobs.

This module provides functionality for generating random workload traces and
jobs for simulation and testing purposes.

Attributes
----------
TRACE_QUANTA : int
    The time interval in seconds for tracing workload utilization.
MAX_NODES_PER_JOB : int
    The maximum number of nodes required for a job.
JOB_NAMES : list
    List of possible job names for random job generation.
CPUS_PER_NODE : int
    Number of CPUs per node.
GPUS_PER_NODE : int
    Number of GPUs per node.
MAX_WALL_TIME : int
    Maximum wall time for a job in seconds.
MIN_WALL_TIME : int
    Minimum wall time for a job in seconds.
JOB_END_PROBS : list
    List of probabilities for different job end states.

"""

import random
import numpy as np

from .job import job_dict

JOB_NAMES = ["LAMMPS", "GROMACS", "VASP", "Quantum ESPRESSO", "NAMD",\
             "OpenFOAM", "WRF", "AMBER", "CP2K", "nek5000", "CHARMM",\
             "ABINIT", "Cactus", "Charm++", "NWChem", "STAR-CCM+",\
             "Gaussian", "ANSYS", "COMSOL", "PLUMED", "nekrs",\
             "TensorFlow", "PyTorch", "BLAST", "Spark", "GAMESS",\
             "ORCA", "Simulink", "MOOSE", "ELK"]

ACCT_NAMES = ["ACT01", "ACT02", "ACT03", "ACT04", "ACT05", "ACT06", "ACT07",\
              "ACT08", "ACT09", "ACT10", "ACT11", "ACT12", "ACT13", "ACT14"]

MAX_PRIORITY = 500000

from .utils import truncated_normalvariate, determine_state, next_arrival


class Workload:
    def __init__(self, *configs):
        """ Initialize Workload with multiple configurations.  """
        self.partitions = [config['system_name'] for config in configs]
        self.config_map = {config['system_name']: config for config in configs}

    def compute_traces(self, cpu_util: float, gpu_util: float, wall_time: int, trace_quanta: int) -> tuple[np.ndarray, np.ndarray]:
        """ Compute CPU and GPU traces based on mean CPU & GPU utilizations and wall time. """
        cpu_trace = cpu_util * np.ones(int(wall_time) // trace_quanta)
        gpu_trace = gpu_util * np.ones(int(wall_time) // trace_quanta)
        return (cpu_trace, gpu_trace)

    def generate_random_jobs(self, num_jobs: int) -> list[list[any]]:
        """ Generate random jobs with specified number of jobs. """
        jobs = []
        for job_index in range(num_jobs):
            # Randomly select a partition
            partition = random.choice(self.partitions)
            # Get the corresponding config for the selected partition
            config = self.config_map[partition]

            nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = random.random() * config['CPUS_PER_NODE']
            gpu_util = random.random() * config['GPUS_PER_NODE']
            mu = (config['MAX_WALL_TIME'] + config['MIN_WALL_TIME']) / 2
            sigma = (config['MAX_WALL_TIME'] - config['MIN_WALL_TIME']) / 6
            wall_time = truncated_normalvariate(mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // 3600 * 3600
            time_limit = truncated_normalvariate(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 3600 * 3600
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []

            # Jobs arrive according to Poisson process
            time_to_next_job = next_arrival(1 / config['JOB_ARRIVAL_TIME'])

            jobs.append(job_dict(nodes_required=nodes_required, name=name,
                                 account=account, cpu_trace=cpu_trace,
                                 gpu_trace=gpu_trace, ntx_trace=net_tx,
                                 nrx_trace=net_rx, end_state=end_state,
                                 id=job_index, priority=priority,
                                 partition=partition,
                                 submit_time=time_to_next_job - 100,
                                 time_limit=time_limit,
                                 start_time=time_to_next_job,
                                 end_time=time_to_next_job + wall_time,
                                 wall_time=wall_time, trace_time=wall_time,
                                 trace_start_time=0, trace_end_time=wall_time))

        return jobs

    def random(self, **kwargs):
        """ Generate random workload """
        num_jobs = kwargs.get('num_jobs', 0)
        return self.generate_random_jobs(num_jobs=num_jobs)

    def peak(self, **kwargs):
        """Peak power test for multiple partitions"""
        jobs = []

        # Iterate through each partition and get its configuration
        for partition in self.partitions:
            # Fetch the config for the current partition
            config = self.config_map[partition]

            # Generate traces based on partition-specific configuration
            cpu_util = config['CPUS_PER_NODE']
            gpu_util = config['GPUS_PER_NODE']
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800, config['TRACE_QUANTA'])
            net_tx, net_rx = [], []

            job_time = len(gpu_trace) * config['TRACE_QUANTA']
            # Create job info for this partition
            job_info = job_dict(
                nodes_required=config['AVAILABLE_NODES'],
                scheduled_nodes=[],  # Down nodes, therefore doesnt work list(range(config['AVAILABLE_NODES'])),
                name=f"Max Test {partition}",
                account=ACCT_NAMES[0],
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                ntx_trace=net_tx,
                nrx_trace=net_rx,
                end_state='COMPLETED',
                id=None,
                priority=100,
                partition=partition,
                time_limit=job_time + 1,
                start_time=0,
                end_time=job_time,
                wall_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time)
            jobs.append(job_info)  # Add job to the list

        return jobs

    def idle(self, **kwargs):
        jobs = []
        # Iterate through each partition and get its configuration
        for partition in self.partitions:
            # Fetch the config for the current partition
            config = self.config_map[partition]

            # Generate traces based on partition-specific configuration
            cpu_util, gpu_util = 0, 0
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800, config['TRACE_QUANTA'])
            net_tx, net_rx = [], []

            job_time = len(gpu_trace) * config['TRACE_QUANTA']
            # Create job info for this partition
            job_info = job_dict(
                nodes_required=config['AVAILABLE_NODES'],
                name=f"Idle Test {partition}",
                account=ACCT_NAMES[0],
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                ntx_trace=net_tx,
                nrx_trace=net_rx,
                end_state='COMPLETED',
                scheduled_nodes=[],  # list(range(config['AVAILABLE_NODES'])),
                id=None,
                priority=100,
                partition=partition,
                time_limit=job_time + 1,
                submit_time=0,
                start_time=0,
                end_time=job_time,
                wall_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time)
            jobs.append(job_info)  # Add job to the list

        return jobs

    def benchmark(self, **kwargs):
        """Benchmark tests for multiple partitions"""

        # List to hold jobs for all partitions
        jobs = []
        account = ACCT_NAMES[0]
        # Iterate through each partition and its config
        for partition in self.partitions:
            # Fetch partition-specific configuration
            config = self.config_map[partition]
            net_tx, net_rx = [], []

            # Max test
            cpu_util, gpu_util = 1, 4
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 10800, config['TRACE_QUANTA'])

            job_time = len(gpu_trace) * config['TRACE_QUANTA']

            job_info = job_dict(
                nodes_required=config['AVAILABLE_NODES'],
                scheduled_nodes=[],  # Explicit scheduled nodes will not work due to down nodes
                name=f"Max Test {partition}",
                account=account,
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                ntx_trace=net_tx,
                nrx_trace=net_rx,
                end_state='COMPLETED',
                id=None,
                priority=100,
                partition=partition,
                submit_time=0,
                time_limit=job_time + 1,
                start_time=0,
                end_time=job_time,
                wall_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False)
            jobs.append(job_info)

            # OpenMxP run
            cpu_util, gpu_util = 0, 4
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600, config['TRACE_QUANTA'])
            job_time = len(gpu_trace) * config['TRACE_QUANTA']

            job_info = job_dict(
                nodes_required=config['AVAILABLE_NODES'],
                scheduled_nodes=[],  # Explicit scheduled nodes will not work due to down nodes
                name=f"OpenMxP {partition}",
                account=account,
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                ntx_trace=net_tx,
                nrx_trace=net_rx,
                end_state='COMPLETED',
                id=None,
                priority=100,
                partition=partition,
                submit_time=0,
                time_limit=job_time + 1,
                start_time=10800,
                end_time=14200,
                wall_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False)
            jobs.append(job_info)

            # HPL run
            cpu_util, gpu_util = 0.33, 0.79 * 4  # based on 24-01-18 run
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600, config['TRACE_QUANTA'])
            job_time = len(gpu_trace) * config['TRACE_QUANTA']
            job_info = job_dict(
                nodes_required=config['AVAILABLE_NODES'],
                scheduled_nodes=[],  # Explicit scheduled nodes will not work due to down nodes
                name=f"HPL {partition}",
                account=account,
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                ntx_trace=net_tx,
                nrx_trace=net_rx,
                end_state='COMPLETED',
                id=None,
                priority=100,
                partition=partition,
                submit_time=0,
                time_limit=job_time + 1,
                start_time=14200,
                end_time=17800,
                wall_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False)
            jobs.append(job_info)

            # Idle test
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, 3600, config['TRACE_QUANTA'])
            job_time = len(gpu_trace) * config['TRACE_QUANTA']
            job_info = job_dict(
                nodes_required=config['AVAILABLE_NODES'],
                scheduled_nodes=[],  # Explicit scheduled nodes will not work due to down nodes
                name=f"Idle Test {partition}",
                account=account,
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                ntx_trace=net_tx,
                nrx_trace=net_rx,
                end_state='COMPLETED',
                id=None,
                priority=100,
                partition=partition,
                submit_time=0,
                time_limit=job_time + 1,
                start_time=17800,
                end_time=21400,
                wall_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False)
            jobs.append(job_info)

        return jobs


    def multitenant(self, **kwargs):
        """
        Generate deterministic jobs to validate multitenant scheduling & power.

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
        mode         = kwargs.get('mode', 'TWO_JOBS_PER_NODE_SPLIT')
        wall_time    = kwargs.get('wall_time', 3600)

        jobs = []

        for partition in self.partitions:
            cfg           = self.config_map[partition]
            trace_quanta  = kwargs.get('trace_quanta', cfg['TRACE_QUANTA'])

            cores_per_cpu = cfg.get('CORES_PER_CPU', 1)
            cpus_per_node = cfg.get('CPUS_PER_NODE', 1)
            cores_per_node = cores_per_cpu * cpus_per_node
            gpus_per_node  = cfg.get('GPUS_PER_NODE', 0)

            n_nodes = cfg['AVAILABLE_NODES']

            def make_trace(cpu_util, gpu_util):
                return self.compute_traces(cpu_util, gpu_util, wall_time, trace_quanta)

            job_id_ctr = 0

            if mode == 'ONE_JOB_PER_NODE_ALL_CORES':
                # Each node runs one job that consumes all cores/GPUs
                for nid in range(n_nodes):
                    cpu_trace, gpu_trace = make_trace(cores_per_node, gpus_per_node)
                    jobs.append(job_dict(
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
                        wall_time=wall_time,
                        trace_time=wall_time,
                        trace_start_time=0,
                        trace_end_time=wall_time
                    ))
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
                        jobs.append(job_dict(
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
                                wall_time=wall_time,
                                trace_time=wall_time,
                                trace_start_time=0,
                                trace_end_time=wall_time
                            ))
                        job_id_ctr += 1

            elif mode == 'STAGGERED_JOBS_PER_NODE':
                # Three jobs per node, staggered starts: 0, wall_time/3, 2*wall_time/3
                offsets = [0, wall_time // 3, 2 * wall_time // 3]
                cpu_each = cores_per_node // 3 or 1
                gpu_each = max(1, gpus_per_node // 3) if gpus_per_node else 0

                for nid in range(n_nodes):
                    for k, offset in enumerate(offsets):
                        cpu_trace, gpu_trace = make_trace(cpu_each, gpu_each)
                        jobs.append(job_dict(
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
                            wall_time=wall_time,
                            trace_time=wall_time,
                            trace_start_time=0,
                            trace_end_time=wall_time
                        ))
                        job_id_ctr += 1
            else:
                raise ValueError(f"Unknown multitenant mode: {mode}")

        return jobs
