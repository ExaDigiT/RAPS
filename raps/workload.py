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
import math
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt

from raps.job import job_dict

JOB_NAMES = ["LAMMPS", "GROMACS", "VASP", "Quantum ESPRESSO", "NAMD",\
             "OpenFOAM", "WRF", "AMBER", "CP2K", "nek5000", "CHARMM",\
             "ABINIT", "Cactus", "Charm++", "NWChem", "STAR-CCM+",\
             "Gaussian", "ANSYS", "COMSOL", "PLUMED", "nekrs",\
             "TensorFlow", "PyTorch", "BLAST", "Spark", "GAMESS",\
             "ORCA", "Simulink", "MOOSE", "ELK"]

ACCT_NAMES = ["ACT01", "ACT02", "ACT03", "ACT04", "ACT05", "ACT06", "ACT07",\
              "ACT08", "ACT09", "ACT10", "ACT11", "ACT12", "ACT13", "ACT14"]

MAX_PRIORITY = 500000

from raps.utils import truncated_normalvariate, determine_state, next_arrival, truncated_weibull


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

    def generate_uniform_jobs(self, *, num_jobs) -> list[list[any]]:
        print("TODO Implement propper!")
        jobs = []
        partition = random.choice(self.partitions)
        config = self.config_map[partition]

        for job_index in range(num_jobs):

            time_to_next_job = next_arrival(1 / config['JOB_ARRIVAL_TIME'])

            nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = random.random() * config['CPUS_PER_NODE']
            gpu_util = random.random() * config['GPUS_PER_NODE']
            mu = config["MIN_WALL_TIME"] * 1.0
            sigma = 4.0
            wall_time = truncated_normalvariate(mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // 3600 * 3600
            time_limit = truncated_normalvariate(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 3600 * 3600
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []
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

    def generate_normal_jobs(self, *, num_jobs) -> list[list[any]]:
        print("TODO Implement propper!")
        jobs = []
        partition = random.choice(self.partitions)
        config = self.config_map[partition]

        for job_index in range(num_jobs):

            time_to_next_job = next_arrival(1 / config['JOB_ARRIVAL_TIME'])

            nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = random.random() * config['CPUS_PER_NODE']
            gpu_util = random.random() * config['GPUS_PER_NODE']
            mu = config["MIN_WALL_TIME"] * 1.0
            sigma = 4.0
            wall_time = truncated_normalvariate(mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // 3600 * 3600
            time_limit = truncated_normalvariate(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 3600 * 3600
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []
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

    def generate_weibull_jobs(self, *, shape, scale, num_jobs) -> list[list[any]]:
        print("TODO Implement propper!")
        jobs = []
        partition = random.choice(self.partitions)
        config = self.config_map[partition]

        for job_index in range(num_jobs):

            time_to_next_job = next_arrival(1 / config['JOB_ARRIVAL_TIME'])

            nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = random.random() * config['CPUS_PER_NODE']
            gpu_util = random.random() * config['GPUS_PER_NODE']
            mu = config["MIN_WALL_TIME"] * 1.0
            sigma = 4.0
            #wall_time = truncated_normalvariate(mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // 3600 * 3600

            wall_time = truncated_weibull(
                    (config['MAX_WALL_TIME'] // 2) + config['MIN_WALL_TIME'], 1,
                    #(config['MAX_WALL_TIME'] // 4) + config['MIN_WALL_TIME'],
                    config['MIN_WALL_TIME'],config['MAX_WALL_TIME']) // 60 * 60  # to 1 minute

                #time_limit = truncated_weibull(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 300 * 300  # to 5 minutes
            time_limit = truncated_weibull(config['MAX_WALL_TIME'] // 2 + config['MIN_WALL_TIME'], 1, wall_time, config['MAX_WALL_TIME']) // 300 * 300  # to 5 minutes


            #time_limit = truncated_normalvariate(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 3600 * 3600
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []
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



    def generate_random_jobs(self, args) -> list[list[any]]:
        """ Generate random jobs with specified number of jobs. """

        partition = random.choice(self.partitions)
        config = self.config_map[partition]
        if args.mu is None:
            mu = (config['MAX_WALL_TIME'] + config['MIN_WALL_TIME']) / 2
        if args.sigma is None:
            sigma = (config['MAX_WALL_TIME'] - config['MIN_WALL_TIME']) / 6

        jobs = []
        for job_index in range(args.numjobs):
            # Randomly select a partition
            # Get the corresponding config for the selected partition
            wes_random = False
            if wes_random:
                nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
                name = random.choice(JOB_NAMES)
                account = random.choice(ACCT_NAMES)
                cpu_util = random.random() * config['CPUS_PER_NODE']
                gpu_util = random.random() * config['GPUS_PER_NODE']
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
                #wall_time = truncated_weibull((config['MAX_WALL_TIME']/4)*3+config['MIN_WALL_TIME'],0.5,config['MIN_WALL_TIME'],config['MAX_WALL_TIME']) // 60 * 60  # to 1 minute
                wall_time = truncated_weibull(
                    (config['MAX_WALL_TIME'] // 2) + config['MIN_WALL_TIME'], 1,
                    #(config['MAX_WALL_TIME'] // 4) + config['MIN_WALL_TIME'],
                    config['MIN_WALL_TIME'],config['MAX_WALL_TIME']) // 60 * 60  # to 1 minute

                #time_limit = truncated_weibull(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 300 * 300  # to 5 minutes
                time_limit = truncated_weibull(config['MAX_WALL_TIME'] // 2 + config['MIN_WALL_TIME'], 1, wall_time, config['MAX_WALL_TIME']) // 300 * 300  # to 5 minutes
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

    def synthetic(self, **kwargs):
        args = kwargs.get('args',None)
        print("ARGS")
        print(args)
        num_jobs = args.numjobs
        #for key,value in kwargs.items():
        #    print(key,value)
        #print("HERE")
        #print(sum(kwargs.get('multimodal')))
        jobs = []
        if len(args.distribution) != 1 and sum(args.multimodal) != 1.0:
            raise Exception(f"Sum of --multimodal != 1.0 : {args.multimodal} == {sum(args.multimodal)}")
        for dist,percentage in zip(args.distribution,args.multimodal):
            print(args.distribution)
            if "uniform" in args.distribution:
                jobs.extend(self.generate_uniform_jobs(num_jobs=int(percentage * num_jobs)))
            elif "weibull" in args.distribution:
                jobs.extend(self.generate_weibull_jobs(shape=args.dist_shape,scale=args.dist_scale,num_jobs=int(percentage * num_jobs)))
            elif "normal" in args.distribution:
                jobs.extend(self.generate_normal_jobs(num_jobs=int(percentage * num_jobs)))
            else:
                pass
        return jobs

    def random(self, **kwargs):
        """ Generate random workload """
        args = kwargs.get('args',None)
        return self.generate_random_jobs(args=args)

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


def plot_job_hist(jobs):

    y = [y['nodes_required'] for y in jobs]
    x = [x['wall_time'] for x in jobs]
    x2 = [x['time_limit'] for x in jobs]
    fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': (4, 1), 'height_ratios': (1, 4)})
    # Remove space between subplots
    fig.subplots_adjust(wspace=0, hspace=0)
    # Create scatter plot
    for i in range(len(x)):
        axs[1,0].plot([x[i],x2[i]],[y[i],y[i]],color='lightblue',zorder=1)
    axs[1, 0].scatter(x2, y,marker='.',c='lightblue',zorder=2)
    axs[1, 0].scatter(x, y,zorder=3)

    axs[0, 0].hist(x2,bins=max(1,min(100,(max(x2) - min(x)))), orientation='vertical',color='lightblue')
    axs[0, 0].hist(x,bins=max(1,min(100,(max(x2) - min(x)))), orientation='vertical')
    #print(x)
    axs[1, 0].sharex(axs[0,0])

    axs[1, 1].hist(y,bins=max(1,min(100,(max(y) - min(y)))), orientation='horizontal')
    axs[1, 0].sharey(axs[1,1])

    # Remove ticks
    axs[0, 0].set_xticks([])
    #axs[0, 0].set_yticks([])
    #axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    #axs[0, 1].set_xticks([])
    #axs[0, 1].set_yticks([])
    #axs[0, 1].set_yticks([])
    axs[0, 1].spines['top'].set_color('white')
    axs[0, 1].set_yticks([])
    axs[0, 1].set_xticks([])
    #axs[0, 1].spines['bottom'].set_color('white')
    #axs[0, 1].spines['left'].set_color('white')
    axs[0, 1].spines['right'].set_color('white')

    axs[1,0].set_ylabel("nodes [N]")
    axs[1,0].set_xlabel("wall time [hh:mm]")
    #axs[1,0].set_yticklabels([str(n).zfill(2) + ':00' for n in np.arange(min(y)//3600, max(y)//3600, 1)])
    minx_s = 0
    maxx_s = max(x2)
    x_label_mins = [n for n in np.arange(minx_s // 60 ,maxx_s // 60 )]
    x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
                            (x1,x2) in [(n // 60,n % 60) for
                                        n in x_label_mins[0::60]]]
    print(x_label_str)
    axs[1,0].set_xticks(x_label_ticks,x_label_str)

    miny = min(y)
    maxy = max(y)
    y_ticks = np.arange(0,maxy,maxy // 10)
    y_ticks[0] = miny
    axs[1,0].set_yticks(y_ticks)

    axs[0,0].tick_params(axis="x", labelbottom=False)
    axs[1,1].tick_params(axis="y", labelleft=False)

    plt.show()


def add_workload_to_parser(parser):

    choices = ['random', 'benchmark', 'peak', 'idle','synthetic']
    parser.add_argument('-w', '--workload', type=str, choices=choices, default=choices[0], help='Type of synthetic workload')

    parser.add_argument("--multimodal", default=[1.0], type=float, nargs="+", help="Percentage to draw from each distribution (list of floats)e.g. '0.2 0.8' percentages apply in order to the list of the  --distribution argument list.")
    parser.add_argument("--distribution", type=str, nargs="+", choices=['uniform','weibull','normal'], default=None, help='Distribution type')
    parser.add_argument("--dist_shape", nargs="+", type=float, required=False, help="Shape of weibull")
    parser.add_argument("--dist_scale", nargs="+", type=float, required=False, help="Scale of weibull")
    parser.add_argument("--mu", nargs="+", type=float, required=False, help="Mean (mu) for Normal distribution")
    parser.add_argument("--sigma", nargs="+", type=float, required=False, help="Standard deviation (sigma) for Normal distribution")

    return parser


if __name__ == "__main__":

    from args import args
    from raps.config import ConfigManager
    config = ConfigManager(system_name=args.system).get_config()

    workload = Workload(config)
    jobs = getattr(workload, args.workload)(args=args)
    plot_job_hist(jobs)
