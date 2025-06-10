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
import matplotlib.pyplot as plt
from raps.telemetry import Telemetry
from raps.job import job_dict
from raps.utils import create_file_indexed, create_dir_indexed, return_nearest_power_of

JOB_NAMES = ["LAMMPS", "GROMACS", "VASP", "Quantum ESPRESSO", "NAMD",\
             "OpenFOAM", "WRF", "AMBER", "CP2K", "nek5000", "CHARMM",\
             "ABINIT", "Cactus", "Charm++", "NWChem", "STAR-CCM+",\
             "Gaussian", "ANSYS", "COMSOL", "PLUMED", "nekrs",\
             "TensorFlow", "PyTorch", "BLAST", "Spark", "GAMESS",\
             "ORCA", "Simulink", "MOOSE", "ELK"]

ACCT_NAMES = ["ACT01", "ACT02", "ACT03", "ACT04", "ACT05", "ACT06", "ACT07",\
              "ACT08", "ACT09", "ACT10", "ACT11", "ACT12", "ACT13", "ACT14"]

MAX_PRIORITY = 500000

from raps.utils import truncated_normalvariate_int, truncated_normalvariate_float, determine_state, next_arrival, next_arrival_byconfargs, truncated_weibull


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

    def job_arrival_distribution_draw_poisson(self,args,config):
        return next_arrival_byconfargs(config,args)

    def job_size_distribution_draw_uniform(self,args,config):
        number = random.randint(1, config['MAX_NODES_PER_JOB'])
        return return_nearest_power_of(number=number, base=args.jobsize_nearest_power_of)

    def job_size_distribution_draw_weibull(self,args,config):
        number = truncated_weibull(args.jobsize_weibull_scale, args.jobsize_weibull_shape, 1, config['MAX_NODES_PER_JOB'])
        return return_nearest_power_of(number=number, base=args.jobsize_nearest_power_of)

    def job_size_distribution_draw_normal(self,args,config):
        number = truncated_normalvariate_int(args.jobsize_normal_mean, args.jobsize_normal_stddev, 1, config['MAX_NODES_PER_JOB'])
        return return_nearest_power_of(number=number, base=args.jobsize_nearest_power_of)

    def cpu_utilization_distribution_draw_uniform(self,args,config):
        return random.uniform(0.0, config['CPUS_PER_NODE'])

    def cpu_utilization_distribution_draw_normal(self,args,config):
        return truncated_normalvariate_float(args.cpuutil_normal_mean, args.cpuutil_normal_stddev,0.0, config['CPUS_PER_NODE'])

    def cpu_utilization_distribution_draw_weibull(self,args,config):
        return truncated_weibull(args.cpuutil_normal_mean, args.cpuutil_normal_stddev,0.0, config['CPUS_PER_NODE'])

    def gpu_utilization_distribution_draw_uniform(self,args,config):
        return random.uniform(0.0, config['GPUS_PER_NODE'])

    def gpu_utilization_distribution_draw_normal(self,args,config):
        return truncated_normalvariate_float(args.gpuutil_normal_mean, args.gpuutil_normal_stddev,0.0, config['GPUS_PER_NODE'])

    def gpu_utilization_distribution_draw_weibull(self,args,config):
        return truncated_weibull(args.gpuutil_normal_mean, args.gpuutil_normal_stddev,0.0, config['GPUS_PER_NODE'])

    def wall_time_distribution_draw_uniform(self,args,config):
        return random.uniform(config['MIN_WALL_TIME'],config['MAX_WALL_TIME'])

    def wall_time_distribution_draw_normal(self,args,config):
        return max(1,truncated_normalvariate_int(float(args.walltime_normal_mean), float(args.walltime_normal_stddev), config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) / 3600 * 3600)

    def wall_time_distribution_draw_weibull(self,args,config):
        return truncated_weibull(args.walltime_weibull_scale, args.walltime_weibull_shape, config['MIN_WALL_TIME'], config['MAX_WALL_TIME'])

    def generate_jobs(self, *,
                      job_arrival_distribution_to_draw_from,
                      job_size_distribution_to_draw_from,
                      cpu_util_distribution_to_draw_from,
                      gpu_util_distribution_to_draw_from,
                      wall_time_distribution_to_draw_from,
                      args
                      ) -> list[list[any]]:
        jobs = []
        partition = random.choice(self.partitions)
        config = self.config_map[partition]
        for job_index in range(args.numjobs):
            submit_time = job_arrival_distribution_to_draw_from(args,config)
            start_time = submit_time
            nodes_required = job_size_distribution_to_draw_from(args,config)
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = cpu_util_distribution_to_draw_from(args,config)
            gpu_util = gpu_util_distribution_to_draw_from(args,config)
            wall_time = wall_time_distribution_to_draw_from(args,config)
            end_time = start_time + wall_time
            time_limit = max(wall_time,wall_time_distribution_to_draw_from(args,config))
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace = cpu_util  # self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            gpu_trace = gpu_util  # self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []
            jobs.append(job_dict(nodes_required=nodes_required, name=name,
                                 account=account, cpu_trace=cpu_trace,
                                 gpu_trace=gpu_trace, ntx_trace=net_tx,
                                 nrx_trace=net_rx, end_state=end_state,
                                 id=job_index, priority=priority,
                                 partition=partition,
                                 submit_time=submit_time,
                                 time_limit=time_limit,
                                 start_time=start_time,
                                 end_time=end_time,
                                 wall_time=wall_time, trace_time=wall_time,
                                 trace_start_time=0, trace_end_time=wall_time))
        return jobs

    def synthetic(self, **kwargs):
        args = kwargs.get('args',None)
        print(args)
        total_jobs = args.numjobs
        orig_job_size_distribution = args.jobsize_distribution
        orig_wall_time_distribution = args.jobsize_distribution
        orig_cpuutil_distribution = args.cpuutil_distribution
        orig_gpuutil_distribution = args.gpuutil_distribution
        jobs = []
        if len(args.jobsize_distribution) != 1 and sum(args.multimodal) != 1.0:
            raise Exception(f"Sum of --multimodal != 1.0 : {args.multimodal} == {sum(args.multimodal)}")
        for i,(jsdist,wtdist,cudist,gudist,percentage) in enumerate(zip(args.jobsize_distribution,
                                                                        args.walltime_distribution,
                                                                        args.cpuutil_distribution,
                                                                        args.gpuutil_distribution,
                                                                        args.multimodal)):

            args.numjobs = math.floor(total_jobs * percentage)
            args.jobsize_distribution = jsdist
            args.walltime_distribution = wtdist
            args.cpuutil_distribution = cudist
            args.gpuutil_distribution = gudist

            job_arrival_distribution_to_draw_from = self.job_arrival_distribution_draw_poisson
            match args.jobsize_distribution:
                case "uniform":
                    job_size_distribution_to_draw_from = self.job_size_distribution_draw_uniform
                case "normal":
                    job_size_distribution_to_draw_from = self.job_size_distribution_draw_normal
                case "weibull":
                    job_size_distribution_to_draw_from = self.job_size_distribution_draw_weibull
                case _:
                    raise NotImplementedError(args.jobsize_distribution)

            match args.walltime_distribution:
                case "weibull":
                    wall_time_distribution_to_draw_from = self.wall_time_distribution_draw_weibull
                case "normal":
                    wall_time_distribution_to_draw_from = self.wall_time_distribution_draw_normal
                case "uniform":
                    wall_time_distribution_to_draw_from = self.wall_time_distribution_draw_uniform
                case _:
                    raise NotImplementedError(args.walltime_distribution)

            match args.cpuutil_distribution:
                case "uniform":
                    cpu_util_distribution_to_draw_from = self.cpu_utilization_distribution_draw_uniform
                case "normal":
                    cpu_util_distribution_to_draw_from = self.cpu_utilization_distribution_draw_normal
                case "weibull":
                    cpu_util_distribution_to_draw_from = self.cpu_utilization_distribution_draw_weibull
                case _:
                    raise NotImplementedError(args.cpuutil_distribution)

            match args.gpuutil_distribution:
                case "uniform":
                    gpu_util_distribution_to_draw_from = self.gpu_utilization_distribution_draw_uniform
                case "normal":
                    gpu_util_distribution_to_draw_from = self.gpu_utilization_distribution_draw_normal
                case "weibull":
                    gpu_util_distribution_to_draw_from = self.gpu_utilization_distribution_draw_weibull
                case _:
                    raise NotImplementedError(args.gpuutil_distribution)

            new_jobs = self.generate_jobs(
                job_arrival_distribution_to_draw_from=job_arrival_distribution_to_draw_from,
                job_size_distribution_to_draw_from=job_size_distribution_to_draw_from,
                cpu_util_distribution_to_draw_from=cpu_util_distribution_to_draw_from,
                gpu_util_distribution_to_draw_from=gpu_util_distribution_to_draw_from,
                wall_time_distribution_to_draw_from=wall_time_distribution_to_draw_from,
                args=args)
            next_arrival(0,reset=True)
            jobs.extend(new_jobs)
        args.numjobs = total_jobs
        args.jobsize_distribution = orig_job_size_distribution
        args.cpuutil_distribution = orig_cpuutil_distribution
        args.gpuutil_distribution = orig_gpuutil_distribution
        args.walltime_distribution = orig_wall_time_distribution
        return jobs

    def generate_random_jobs(self, args) -> list[list[any]]:
        """ Generate random jobs with specified number of jobs. """

        partition = random.choice(self.partitions)
        config = self.config_map[partition]

        jobs = []
        for job_index in range(args.numjobs):
            # Randomly select a partition
            # Get the corresponding config for the selected partition
            nodes_required = random.randint(1, config['MAX_NODES_PER_JOB'])
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = random.random() * config['CPUS_PER_NODE']
            gpu_util = random.random() * config['GPUS_PER_NODE']
            mu = (config['MAX_WALL_TIME'] + config['MIN_WALL_TIME']) / 2
            sigma = (config['MAX_WALL_TIME'] - config['MIN_WALL_TIME']) / 6
            wall_time = truncated_normalvariate_int(mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // 3600 * 3600
            time_limit = truncated_normalvariate_int(mu, sigma, wall_time, config['MAX_WALL_TIME']) // 3600 * 3600
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = [], []

            # Jobs arrive according to Poisson process
            time_to_next_job = next_arrival_byconfargs(config,args)

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


def plot_job_hist(jobs,config=None,dist_split=None):
    # put args.multimodal in dist_split!
    split = [1.0]
    num_dist = 1
    if dist_split:
        num_dist = len(dist_split)
        split = dist_split

    y = [y['nodes_required'] for y in jobs]
    x = [x['wall_time'] for x in jobs]
    x2 = [x['time_limit'] for x in jobs]
    fig_m = plt.figure()
    gs = fig_m.add_gridspec(30, 1)
    gs0 = gs[0:20].subgridspec(500,500,hspace=0,wspace=0)
    gs1 = gs[24:].subgridspec(1,1)

    ax_top = fig_m.add_subplot(gs0[:])
    ax_top.axis('off')
    ax_top.set_title('Job Distribution')

    ax_bot = fig_m.add_subplot(gs1[:])
    ax_bot.axis('off')
    ax_bot.set_title('Submit Time + Wall Time')

    #ax0 = fig_m.add_subplot(gs[:2,:])
    #ax1 = fig_m.add_subplot(gs[2:,:])

    #gss = gridspec.GridSpec(5, 5, figure=ax0)
    #fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': (4, 1), 'height_ratios': (1, 4)})
    axs = []
    col = []
    col.append(fig_m.add_subplot(gs0[:100,:433]))
    col.append(fig_m.add_subplot(gs0[:100,433:]))
    axs.append(col.copy())
    col = []
    col.append(fig_m.add_subplot(gs0[100:,:433]))
    col.append(fig_m.add_subplot(gs0[100:,433:]))
    axs.append(col.copy())

    ax_b = fig_m.add_subplot(gs1[:,:])

    # Create scatter plot
    for i in range(len(x)):
        axs[1][0].plot([x[i],x2[i]],[y[i],y[i]],color='lightblue',zorder=1)
    axs[1][0].scatter(x2, y,marker='.',c='lightblue',zorder=2)
    axs[1][0].scatter(x, y,zorder=3)

    cpu_util = [x['cpu_trace'] for x in jobs]
    if isinstance(cpu_util[0],np.ndarray):
        cpu_util = np.concatenate(cpu_util).ravel()
    gpu_util = [x['gpu_trace'] for x in jobs]
    if isinstance(gpu_util[0],np.ndarray):
        gpu_util = np.concatenate(gpu_util).ravel()
    if not all([x == 0 for x in gpu_util]):
        axs[0][1].scatter(cpu_util,gpu_util,zorder=2,marker='.',s=0.2)
        axs[0][1].hist(gpu_util,bins=100,orientation='horizontal',zorder=1, density=True,color='tab:purple')
        axs[0][1].axhline(np.mean(gpu_util), color='r', linewidth=1,zorder=3)
        axs[0][1].set(ylim=[0,config['GPUS_PER_NODE']])
        axs[0][1].set_ylabel("gpu util")
        axs[0][1].yaxis.set_label_coords(1.15, 0.5)
        axs[0][1].yaxis.set_label_position("right")
        axs[0][1].yaxis.tick_right()
    else:
        axs[0][1].set_yticks([])
        pass
    axs[0][1].hist(cpu_util,bins=100,orientation='vertical',zorder=1, density=True,color='tab:cyan')
    axs[0][1].axvline(np.mean(cpu_util), color='r', linewidth=1,zorder=3)
    axs[0][1].set(xlim=[0,config['CPUS_PER_NODE']])
    axs[0][1].set_xlabel("cpu util")
    axs[0][1].xaxis.set_label_coords(0.5,1.30)
    axs[0][1].xaxis.set_label_position("top")
    axs[0][1].xaxis.tick_top()
    axs[0][0].hist(x2,bins=max(1,math.ceil(min(100,(max(x2) - min(x))))), orientation='vertical',color='lightblue')
    axs[0][0].hist(x,bins=max(1,math.ceil(min(100,(max(x2) - min(x))))), orientation='vertical')
    axs[1][0].sharex(axs[0][0])

    axs[1][1].hist(y,bins=max(1,min(100,(max(y) - min(y)))), orientation='horizontal')
    axs[1][0].sharey(axs[1][1])

    # Remove ticks
    axs[0][0].set_xticks([])
    axs[1][1].set_yticks([])
    axs[0][1].spines['top'].set_color('white')
    axs[0][1].spines['right'].set_color('white')
    axs[1][0].set_ylabel("nodes [N]")
    axs[1][0].set_xlabel("wall time [hh:mm]")
    minx_s = 0
    maxx_s = math.ceil(max(x2))
    x_label_mins = [n for n in np.arange(minx_s // 60, maxx_s // 60)]
    x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
                            (x1,x2) in [(n // 60,n % 60) for
                                        n in x_label_mins[0::60]]]
    axs[1][0].set_xticks(x_label_ticks,x_label_str)
    miny = min(y)
    maxy = max(y)
    interval = max(1,maxy // 10)
    y_ticks = np.arange(0, maxy, interval)
    y_ticks[0] = miny
    axs[1][0].set_yticks(y_ticks)

    axs[0][0].tick_params(axis="x", labelbottom=False)
    axs[1][1].tick_params(axis="y", labelleft=False)

    # Submit_time and Wall_time
    duration = [x['wall_time'] for x in jobs]
    nodes_required = [x['nodes_required'] for x in jobs]
    submit_t = [x['submit_time'] for x in jobs]

    offset = 0
    split_index = 0
    split_offset = math.floor(len(x) * split[split_index])
    gantt_nodes = args.gantt_nodes
    if gantt_nodes:
        if split[0] == 0.0:
            ax_b.axhline(y=offset, color='red', linestyle='--',lw=0.5)
            split_index += 1
        for i in range(len(x)):
            #ax_b.barh(i,duration[i], height=1.0, left=submit_t[i])
            ax_b.barh(offset + nodes_required[i] / 2,duration[i], height=nodes_required[i], left=submit_t[i])
            offset += nodes_required[i]
            if i != len(x) - 1 and i == split_offset - 1 and split_index < len(split):
                ax_b.axhline(y=offset, color='red', linestyle='--',lw=0.5)
                split_index += 1
                split_offset += math.floor(len(x) * split[split_index])
                #ax_b.axhline(y=(len(x)/num_dist * i)-0.5, color='red', linestyle='--',lw=0.5)
        if split[-1] == 0.0:
            ax_b.axhline(y=offset, color='red', linestyle='--',lw=0.5)
            split_index += 1
        ax_b.set_ylabel("Jobs' acc. nodes")
    else:
        for i in range(len(x)):
            ax_b.barh(i,duration[i], height=1.0, left=submit_t[i])
        for i in range(1,num_dist):
            if num_dist == 1:
                break
            ax_b.axhline(y=(len(x) * split[split_index]) - 0.5, color='red', linestyle='--',lw=0.5)
            split_index += 1
        ax_b.set_ylabel("Job ID")
        #ax_b labels:
    ax_b.set_xlabel("time [hh:mm]")
    minx_s = 0
    maxx_s = math.ceil(max([x['wall_time'] for x in jobs]) + max([x['submit_time'] for x in jobs]))
    x_label_mins = [n for n in np.arange(minx_s // 60, maxx_s // 60)]
    x_label_ticks = [n * 60 for n in x_label_mins[0::60]]
    x_label_str = [str(x1).zfill(2) + ":" + str(x2).zfill(2) for
                            (x1,x2) in [(n // 60,n % 60) for
                                        n in x_label_mins[0::60]]]

    ax_b.set_xticks(x_label_ticks,x_label_str)
    ax_b.yaxis.set_inverted(True)

    plt.show()


def add_workload_to_parser(parser):

    choices = ['random', 'benchmark', 'peak', 'idle','synthetic']
    parser.add_argument('-w', '--workload', type=str, choices=choices, default=choices[0], help='Type of synthetic workload')

    parser.add_argument("--multimodal", default=[1.0], type=float, nargs="+", help="Percentage to draw from each distribution (list of floats)e.g. '0.2 0.8' percentages apply in order to the list of the  --distribution argument list.")

    # Jobsize:
    parser.add_argument("--jobsize-distribution", type=str, nargs="+", choices=['uniform','weibull','normal'], default=None, help='Distribution type')

    parser.add_argument("--jobsize-normal-mean", type=float, required=False, help="Mean (mu) for Normal distribution")
    parser.add_argument("--jobsize-normal-stddev", type=float, required=False, help="Standard deviation (sigma) for Normal distribution")

    parser.add_argument("--jobsize-weibull-shape", type=float, required=False, help="Jobsize shape of weibull")
    parser.add_argument("--jobsize-weibull-scale", type=float, required=False, help="Jobsize scale of weibull")

    parser.add_argument("--jobsize-nearest-power-of", default=1, type=int,required=False,help="Map random samples to the nearest power of N your choice. (Experimental: This changes the shape of the distribution, as density of powers change on the numberline!)")

    # Walltime:
    parser.add_argument("--walltime-distribution", type=str, nargs="+", choices=['uniform','weibull','normal'], default=None, help='Distribution type')

    parser.add_argument("--walltime-normal-mean", type=float, required=False, help="Walltime mean (mu) for Normal distribution")
    parser.add_argument("--walltime-normal-stddev", type=float, required=False, help="Walltime standard deviation (sigma) for Normal distribution")

    parser.add_argument("--walltime-weibull-shape", type=float, required=False, help="Walltime shape of weibull")
    parser.add_argument("--walltime-weibull-scale", type=float, required=False, help="Walltime scale of weibull")

    # Utilizations
    parser.add_argument("--cpuutil-distribution", type=str, nargs="+", choices=['uniform','weibull','normal'], default=['uniform'], help='Distribution type')

    parser.add_argument("--cpuutil-normal-mean", type=float, required=False, help="Walltime mean (mu) for Normal distribution")
    parser.add_argument("--cpuutil-normal-stddev", type=float, required=False, help="Walltime standard deviation (sigma) for Normal distribution")

    parser.add_argument("--cpuutil-weibull-shape", type=float, required=False, help="Walltime shape of weibull")
    parser.add_argument("--cpuutil-weibull-scale", type=float, required=False, help="Walltime scale of weibull")

    parser.add_argument("--gpuutil-distribution", type=str, nargs="+", choices=['uniform','weibull','normal'], default=['uniform'], help='Distribution type')

    parser.add_argument("--gpuutil-normal-mean", type=float, required=False, help="Walltime mean (mu) for Normal distribution")
    parser.add_argument("--gpuutil-normal-stddev", type=float, required=False, help="Walltime standard deviation (sigma) for Normal distribution")

    parser.add_argument("--gpuutil-weibull-shape", type=float, required=False, help="Walltime shape of weibull")
    parser.add_argument("--gpuutil-weibull-scale", type=float, required=False, help="Walltime scale of weibull")

    parser.add_argument("--gantt-nodes", default=False, action='store_true', required=False, help="Print Gannt with nodes required as line thickness (default false)")

    return parser


if __name__ == "__main__":

    from args import args, args_dict
    from raps.config import ConfigManager
    config = ConfigManager(system_name=args.system).get_config()
    if args.replay:
        td = Telemetry(**args_dict)
        jobs,_,_,_ = td.load_jobs_times_args_from_files(files=args.replay,args=args)
    else:
        workload = Workload(config)
        jobs = getattr(workload, args.workload)(args=args)
    plot_job_hist(jobs, config=config, dist_split=args.multimodal)
    if args.output:
        filename = create_file_indexed('wl',create=False)
        timestep_start = min([x['submit_time'] for x in jobs])
        timestep_end = math.ceil(max([x['submit_time'] for x in jobs]) + max([x['wall_time'] for x in jobs]))
        np.savez_compressed(filename,jobs=jobs,timestep_start=timestep_start, timestep_end=timestep_end, args=args)
