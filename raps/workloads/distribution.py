import math
import random

from raps.job import Job, job_dict
from raps.utils import (
    truncated_normalvariate_int,
    truncated_normalvariate_float,
    truncated_weibull,
    truncated_weibull_float,
    determine_state,
    next_arrival_byconfargs,
)

from .constants import JOB_NAMES, ACCT_NAMES, MAX_PRIORITY

class DistributionWorkload:

    def job_arrival_distribution_draw_poisson(self, args, config):
        return next_arrival_byconfargs(config, args)


    def job_size_distribution_draw_uniform(self, args, config):
        min_v = 1
        max_v = config['MAX_NODES_PER_JOB']
        if (args.jobsize_is_power_of is not None):
            base = args.jobsize_is_power_of
            possible_jobsizes = [base ** exp for exp in range(min_v, int(math.floor(math.log(max_v, base))))]
            selection = random.randint(0, len(possible_jobsizes) - 1)
            number = possible_jobsizes[selection]
        elif (args.jobsize_is_of_degree is not None):
            exp = args.jobsize_is_of_degree
            possible_jobsizes = [base ** exp for base in range(min_v, int(math.floor(pow(max_v, 1 / exp))))]
            selection = random.randint(0, len(possible_jobsizes) - 1)
            number = possible_jobsizes[selection]
        else:
            number = random.randint(1, config['MAX_NODES_PER_JOB'])
        return number


    def job_size_distribution_draw_weibull(self, args, config):
        min_v = 1
        max_v = config['MAX_NODES_PER_JOB']
        if (args.jobsize_is_power_of is not None):
            base = args.jobsize_is_power_of
            possible_jobsizes = [base ** exp for exp in range(min_v, int(math.floor(math.log(max_v, base))))]
            scale = math.log(args.jobsize_weibull_scale, base)
            shape = math.log(args.jobsize_weibull_shape, base)
            selection = truncated_weibull(scale, shape, 0, len(possible_jobsizes) - 1)
            number = possible_jobsizes[selection]
        elif (args.jobsize_is_of_degree is not None):
            exp = args.jobsize_is_of_degree
            possible_jobsizes = [base ** exp for base in range(min_v, int(math.floor(pow(max_v, 1 / exp))))]
            scale = math.pow(args.jobsize_weibull_scale, 1 / exp)
            shape = math.pow(args.jobsize_weibull_shape, 1 / exp)
            selection = truncated_weibull(scale, shape, 0, len(possible_jobsizes) - 1)
            number = possible_jobsizes[selection]
        else:
            number = truncated_weibull(args.jobsize_weibull_scale, args.jobsize_weibull_shape,
                                       1, config['MAX_NODES_PER_JOB'])
        return number


    def job_size_distribution_draw_normal(self, args, config):
        min_v = 1
        max_v = config['MAX_NODES_PER_JOB']
        if (args.jobsize_is_power_of is not None):
            base = args.jobsize_is_power_of
            possible_jobsizes = [base ** exp for exp in range(min_v, int(math.floor(math.log(max_v, base))))]
            mean = math.log(args.jobsize_normal_mean, base)
            stddev = math.log(args.jobsize_normal_stddev, base)  # (len(possible_jobsizes) / (max_v - min_v))
            selection = truncated_normalvariate_int(mean, stddev, 0, len(possible_jobsizes) - 1)
            number = possible_jobsizes[selection - 1]
        elif (args.jobsize_is_of_degree is not None):
            exp = args.jobsize_is_of_degree
            possible_jobsizes = [base ** exp for base in range(min_v, int(math.floor(pow(max_v, 1 / exp))))]
            mean = math.pow(args.jobsize_normal_mean, 1 / exp)
            stddev = math.pow(args.jobsize_normal_stddev, 1 / exp)
            selection = truncated_weibull(mean, stddev, 0, len(possible_jobsizes) - 1)
            number = possible_jobsizes[selection]
        else:
            number = truncated_normalvariate_int(
                args.jobsize_normal_mean, args.jobsize_normal_stddev, 1, config['MAX_NODES_PER_JOB'])
        return number


    def cpu_utilization_distribution_draw_uniform(self, args, config):
        return random.uniform(0.0, config['CPUS_PER_NODE'])


    def cpu_utilization_distribution_draw_normal(self, args, config):
        return truncated_normalvariate_float(args.cpuutil_normal_mean,
                                             args.cpuutil_normal_stddev,
                                             0.0, config['CPUS_PER_NODE'])


    def cpu_utilization_distribution_draw_weibull(self, args, config):
        return truncated_weibull_float(args.cpuutil_weibull_scale,
                                       args.cpuutil_weibull_shape,
                                       0.0, config['CPUS_PER_NODE'])


    def gpu_utilization_distribution_draw_uniform(self, args, config):
        return random.uniform(0.0, config['GPUS_PER_NODE'])


    def gpu_utilization_distribution_draw_normal(self, args, config):
        return truncated_normalvariate_float(args.gpuutil_normal_mean,
                                             args.gpuutil_normal_stddev,
                                             0.0, config['GPUS_PER_NODE'])


    def gpu_utilization_distribution_draw_weibull(self, args, config):
        return truncated_weibull_float(args.gpuutil_weibull_scale,
                                       args.gpuutil_weibull_shape,
                                       0.0, config['GPUS_PER_NODE'])


    def wall_time_distribution_draw_uniform(self, args, config):
        return random.uniform(config['MIN_WALL_TIME'], config['MAX_WALL_TIME'])


    def wall_time_distribution_draw_normal(self, args, config):
        return max(1, truncated_normalvariate_int(float(args.walltime_normal_mean),
                   float(args.walltime_normal_stddev), config['MIN_WALL_TIME'],
                   config['MAX_WALL_TIME']) / 3600 * 3600)


    def wall_time_distribution_draw_weibull(self, args, config):
        return truncated_weibull(args.walltime_weibull_scale,
                                 args.walltime_weibull_shape,
                                 config['MIN_WALL_TIME'], config['MAX_WALL_TIME'])


    def generate_jobs_from_distribution(self, *,
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
            submit_time = int(job_arrival_distribution_to_draw_from(args, config))
            start_time = submit_time
            nodes_required = job_size_distribution_to_draw_from(args, config)
            name = random.choice(JOB_NAMES)
            account = random.choice(ACCT_NAMES)
            cpu_util = cpu_util_distribution_to_draw_from(args, config)
            if "CORES_PER_CPU" in config:
                cpu_cores_required = random.randint(0, config["CORES_PER_CPU"])
            else:
                cpu_cores_required = None
            gpu_util = gpu_util_distribution_to_draw_from(args, config)
            if "GPUS_PER_NODE" in config:
                if isinstance(gpu_util, list):
                    gpu_units_required = random.randint(0, max(config["GPUS_PER_NODE"], math.ceil(max(gpu_util))))
                else:
                    gpu_units_required = random.randint(0, max(config["GPUS_PER_NODE"], math.ceil(gpu_util)))
            wall_time = wall_time_distribution_to_draw_from(args, config)
            end_time = start_time + wall_time
            time_limit = max(wall_time, wall_time_distribution_to_draw_from(args, config))
            end_state = determine_state(config['JOB_END_PROBS'])
            cpu_trace = cpu_util  # self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            gpu_trace = gpu_util  # self.compute_traces(cpu_util, gpu_util, wall_time, config['TRACE_QUANTA'])
            priority = random.randint(0, MAX_PRIORITY)
            net_tx, net_rx = None, None
            job_info = job_dict(nodes_required=nodes_required, name=name,
                                account=account, cpu_trace=cpu_trace,
                                gpu_trace=gpu_trace, ntx_trace=net_tx,
                                nrx_trace=net_rx, end_state=end_state,
                                id=job_index, priority=priority,
                                partition=partition,
                                submit_time=submit_time,
                                time_limit=time_limit,
                                start_time=start_time,
                                end_time=end_time,
                                expected_run_time=wall_time, trace_time=wall_time,
                                trace_start_time=0, trace_end_time=wall_time,
                                cpu_cores_required=cpu_cores_required,
                                gpu_units_required=gpu_units_required,
                                trace_quanta=config['TRACE_QUANTA']
                                )
            job = Job(job_info)
            jobs.append(job)
        return jobs
