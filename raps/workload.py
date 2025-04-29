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

from .utils import truncated_normalvariate, determine_state, next_arrival, truncated_weibull


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
            wes_random = False
            if wes_random:
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
            else:
                max_nodes = config['MAX_NODES_PER_JOB']
                min_nodes = 1
                nodes_required = truncated_weibull(max_nodes, 0.1, min_nodes, max_nodes)
                name = random.choice(JOB_NAMES)
                account = random.choice(ACCT_NAMES)
                cpu_util = random.random() * config['CPUS_PER_NODE']
                gpu_util = random.random() * config['GPUS_PER_NODE']
                mu = (config['MAX_WALL_TIME'] + config['MIN_WALL_TIME']) / 2
                sigma = (config['MAX_WALL_TIME'] - config['MIN_WALL_TIME']) / 6
                wall_time = truncated_weibull(3 * config['MIN_WALL_TIME'],0.75,config['MIN_WALL_TIME'],config['MAX_WALL_TIME']) // 60 * 60  # to 1 minute
                time_limit = truncated_normalvariate(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 300 * 300  # to 5 minutes
                end_state = determine_state(config['JOB_END_PROBS'])
                cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
                if nodes_required < max_nodes * .10:
                    priority = 0
                elif nodes_required < max_nodes * .20:
                    priority = 1
                elif nodes_required < max_nodes * .50:
                    priority = 2
                else:
                    priority = 3
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
