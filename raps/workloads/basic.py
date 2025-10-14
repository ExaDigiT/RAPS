import math
import random

from raps.job import Job, job_dict
from raps.utils import (
    truncated_normalvariate_int,
    determine_state,
    next_arrival,
    next_arrival_byconfargs,
)

from .constants import JOB_NAMES, ACCT_NAMES, MAX_PRIORITY


class BasicWorkload:

    # Test for random 'reasonable' AI jobs
    def randomAI(self, **kwargs):
        args = kwargs.get('args', None)
        jobs = []
        for i in range(args.numjobs):
            draw = random.randint(0, 10)
            if draw != 0:
                et = random.randint(7200, 28800)
                nr = random.choice([128, 256, 512, 1024, 1280, 1792, 2048])
                new_job = Job(job_dict(nodes_required=nr,
                                       name="LLM Production",
                                       account="llmUser",
                                       end_state="Success",
                                       id=random.randint(1, 99999),
                                       cpu_trace=0.1,
                                       gpu_trace=(random.uniform(0.55, 0.8)
                                                  * self.config_map[self.args.system]['GPUS_PER_NODE']),
                                       ntx_trace=None,
                                       nrx_trace=None,
                                       submit_time=0,
                                       time_limit=random.randint(43200, 43200),
                                       start_time=0,
                                       end_time=et,
                                       expected_run_time=et))
            else:
                et = random.randint(300, 7200)
                nr = random.choice([1, 1, 1, 1, 1, 2, 4, 8, 16, 32, 128])
                new_job = Job(job_dict(nodes_required=nr,
                                       name="User-Test LLM",
                                       account="llmUser",
                                       end_state="Success",
                                       id=random.randint(1, 99999),
                                       cpu_trace=1,
                                       gpu_trace=(0.2 * self.config_map[self.args.system]['GPUS_PER_NODE']),
                                       ntx_trace=None,
                                       nrx_trace=None,
                                       submit_time=0,
                                       time_limit=43200,
                                       start_time=0,
                                       end_time=et,
                                       expected_run_time=random.randint(60, 7200)))
            jobs.append(new_job)
        return jobs

    def synthetic(self, **kwargs):
        args = kwargs.get('args', None)
        print(args)
        total_jobs = args.numjobs
        orig_job_size_distribution = args.jobsize_distribution
        orig_wall_time_distribution = args.walltime_distribution
        orig_cpuutil_distribution = args.cpuutil_distribution
        orig_gpuutil_distribution = args.gpuutil_distribution
        jobs = []
        if len(args.jobsize_distribution) != 1 and sum(args.multimodal) != 1.0:
            raise Exception(f"Sum of --multimodal != 1.0 : {args.multimodal} == {sum(args.multimodal)}")
        for i, (jsdist, wtdist, cudist, gudist, percentage) in enumerate(zip(args.jobsize_distribution,
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

            new_jobs = self.generate_jobs_from_distribution(
                job_arrival_distribution_to_draw_from=job_arrival_distribution_to_draw_from,
                job_size_distribution_to_draw_from=job_size_distribution_to_draw_from,
                cpu_util_distribution_to_draw_from=cpu_util_distribution_to_draw_from,
                gpu_util_distribution_to_draw_from=gpu_util_distribution_to_draw_from,
                wall_time_distribution_to_draw_from=wall_time_distribution_to_draw_from,
                args=args)
            next_arrival(0, reset=True)
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

        # time_delta = args.time_delta  # Unused
        downscale = args.downscale

        config['MIN_WALL_TIME'] = config['MIN_WALL_TIME'] * downscale
        config['MAX_WALL_TIME'] = config['MAX_WALL_TIME'] * downscale
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
            wall_time = (truncated_normalvariate_int(
                mu, sigma, config['MIN_WALL_TIME'], config['MAX_WALL_TIME']) // (3600 * downscale) * (3600 * downscale))
            time_limit = (truncated_normalvariate_int(mu, sigma, wall_time,
                          config['MAX_WALL_TIME']) // (3600 * downscale) * (3600 * downscale))
            # print(f"wall_time: {wall_time//downscale}")
            # print(f"time_limit: {time_limit//downscale}")
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace, gpu_trace = self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = None, None

            # Jobs arrive according to Poisson process
            time_to_next_job = int(next_arrival_byconfargs(config, args))
            # wall_time = wall_time * downscale
            # time_limit = time_limit * downscale

            job_info = job_dict(nodes_required=nodes_required, name=name,
                                account=account, cpu_trace=cpu_trace,
                                gpu_trace=gpu_trace, ntx_trace=net_tx,
                                nrx_trace=net_rx, end_state=end_state,
                                id=job_index, priority=priority,
                                partition=partition,
                                submit_time=time_to_next_job - 100,
                                time_limit=time_limit,
                                start_time=time_to_next_job,
                                end_time=time_to_next_job + wall_time,
                                expected_run_time=wall_time, trace_time=wall_time,
                                trace_start_time=0, trace_end_time=wall_time,
                                trace_quanta=config['TRACE_QUANTA'] * downscale,
                                downscale=downscale
                                )
            job = Job(job_info)
            jobs.append(job)
        return jobs

    def random(self, **kwargs):
        """ Generate random workload """
        args = kwargs.get('args', None)
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
            net_tx, net_rx = None, None

            job_time = len(gpu_trace) * config['TRACE_QUANTA']
            # Create job info for this partition
            job_info = job_dict(nodes_required=config['AVAILABLE_NODES'],
                                # Down nodes, therefore doesnt work list(range(config['AVAILABLE_NODES'])),
                                scheduled_nodes=[],
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
                                expected_run_time=job_time,
                                trace_time=job_time,
                                trace_start_time=0,
                                trace_end_time=job_time,
                                trace_quanta=config['TRACE_QUANTA']
                                )
            job = Job(job_info)
            jobs.append(job)  # Add job to the list

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
            net_tx, net_rx = None, None

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
                expected_run_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_quanta=config['TRACE_QUANTA'])
            job = Job(job_info)
            jobs.append(job)  # Add job to the list

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
            net_tx, net_rx = None, None

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
                expected_run_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False,
                trace_quanta=config['TRACE_QUANTA'])
            job = Job(job_info)
            jobs.append(job)

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
                expected_run_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False,
                trace_quanta=config['TRACE_QUANTA'])
            job = Job(job_info)
            jobs.append(job)

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
                expected_run_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False,
                trace_quanta=config['TRACE_QUANTA'])
            job = Job(job_info)
            jobs.append(job)

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
                expected_run_time=job_time,
                trace_time=job_time,
                trace_start_time=0,
                trace_end_time=job_time,
                trace_missing_values=False,
                trace_quanta=config['TRACE_QUANTA'])
            job = Job(job_info)
            jobs.append(job)

        return jobs
