"""
This module provides functionality for generating statistics.
These are statistics on
the engine
the jobs

Both could be part of the engine or jobs class, but as the are very verbose,
try to keep statistics consolidated in this file.
"""
import sys
from .utils import sum_values, min_value, max_value, convert_seconds_to_hhmmss

from .engine import Engine


def get_engine_stats(engine: Engine):
    """ Return engine statistics """
    timesteps = engine.current_timestep - engine.timestep_start
    num_samples = len(engine.power_manager.history) if engine.power_manager else 0
    time_simulated = convert_seconds_to_hhmmss(timesteps / engine.downscale)
    average_power_mw = sum_values(engine.power_manager.history) / num_samples / 1000 if num_samples else 0
    average_loss_mw = sum_values(engine.power_manager.loss_history) / num_samples / 1000 if num_samples else 0
    min_loss_mw = min_value(engine.power_manager.loss_history) / 1000 if num_samples else 0
    max_loss_mw = max_value(engine.power_manager.loss_history) / 1000 if num_samples else 0

    loss_fraction = average_loss_mw / average_power_mw if average_power_mw else 0
    efficiency = 1 - loss_fraction if loss_fraction else 0
    total_energy_consumed = average_power_mw * timesteps / 3600 if timesteps else 0  # MW-hr
    emissions = total_energy_consumed * 852.3 / 2204.6 / efficiency if efficiency else 0
    total_cost = total_energy_consumed * 1000 * engine.config.get('POWER_COST', 0)  # Total cost in dollars

    stats = {
        'time simulated': time_simulated,
        'num_samples': num_samples,
        'average power': f'{average_power_mw:.2f} MW',
        'min loss': f'{min_loss_mw:.2f} MW',
        'average loss': f'{average_loss_mw:.2f} MW',
        'max loss': f'{max_loss_mw:.2f} MW',
        'system power efficiency': f'{efficiency * 100:.2f}%',
        'total energy consumed': f'{total_energy_consumed:.2f} MW-hr',
        'carbon emissions': f'{emissions:.2f} metric tons CO2',
        'total cost': f'${total_cost:.2f}'
    }

    if engine.config['multitenant']:
        # Multitenancy Stats
        total_jobs_loaded = engine.total_initial_jobs  # Assuming this is passed to __init__
        stats['total jobs loaded'] = total_jobs_loaded
        stats['jobs completed percentage'] = f"{(engine.jobs_completed / total_jobs_loaded * 100):.2f}%"

    if engine.node_occupancy_history:
        # Calculate average concurrent jobs per node (average density across all nodes and timesteps)
        total_jobs_running_timesteps = 0
        max_concurrent_jobs_per_node = 0
        sum_jobs_per_active_node = 0  # New: Sum of (jobs / active_nodes) for each timestep
        count_active_timesteps_for_avg_active = 0  # New: Count of timesteps with active nodes

        for occupancy_dict in engine.node_occupancy_history:
            current_timestep_total_occupancy = sum(occupancy_dict.values())
            total_jobs_running_timesteps += current_timestep_total_occupancy

            # Find max concurrent jobs on any single node for this timestep
            if occupancy_dict:
                max_concurrent_jobs_per_node = max(max_concurrent_jobs_per_node, max(occupancy_dict.values()))

            # New: Calculate average jobs per *active* node for this timestep
            active_nodes_in_timestep = [count for count in occupancy_dict.values() if count > 0]
            if active_nodes_in_timestep:
                sum_jobs_per_active_node += sum(active_nodes_in_timestep) / len(active_nodes_in_timestep)
                count_active_timesteps_for_avg_active += 1

        # Average jobs per *active* node (user's desired "1" type)
        avg_jobs_per_active_node = (sum_jobs_per_active_node / count_active_timesteps_for_avg_active) \
            if count_active_timesteps_for_avg_active > 0 else 0

        stats['avg concurrent jobs per active node'] = f"{avg_jobs_per_active_node:.2f}"
        stats['max concurrent jobs per node'] = max_concurrent_jobs_per_node
    else:
        stats['avg concurrent jobs per node'] = "N/A"
        stats['max concurrent jobs per node'] = "N/A"

    # network_stats = get_network_stats()
    # stats.update(network_stats)

    return stats


def min_max_sum(value, min, max, sum):
    if value < 0:
        value = 0
    if value < min:
        min = value
    if value > max:
        max = value
    sum += value
    return min, max, sum


def get_scheduler_stats(engine: Engine):
    if len(engine.scheduler_queue_history) != 0:
        average_queue = sum(engine.scheduler_queue_history) / len(engine.scheduler_queue_history)
    else:
        average_queue = 0
    if len(engine.scheduler_running_history) != 0:
        average_running = sum(engine.scheduler_running_history) / len(engine.scheduler_running_history)
    else:
        average_running = 0

    stats = {
        'average_queue': average_queue,
        'average_running': average_running,
    }
    return stats


def get_network_stats(engine: Engine):
    stats = {}

    if engine.net_util_history:
        mean_net_util = sum(engine.net_util_history) / len(engine.net_util_history)
    else:
        mean_net_util = 0.0

    stats["avg network util"] = f"{mean_net_util * 100:.2f}%"

    if engine.avg_slowdown_history:
        avg_job_slow = sum(engine.avg_slowdown_history) / len(engine.avg_slowdown_history)
    else:
        avg_job_slow = 1.0
    stats["avg per-job slowdown"] = f"{avg_job_slow:.2f}x"

    if engine.max_slowdown_history:
        max_job_slow = max(engine.max_slowdown_history)
    else:
        max_job_slow = 1.0
    stats["max per-job slowdown"] = f"{max_job_slow:.2f}x"

    return stats


def get_job_stats(engine: Engine):
    """ Return job statistics processed over the engine execution"""
    # Information on Job-Mix
    min_job_size, max_job_size, sum_job_size = sys.maxsize, -sys.maxsize - 1, 0
    min_runtime, max_runtime, sum_runtime = sys.maxsize, -sys.maxsize - 1, 0

    min_energy, max_energy, sum_energy = sys.maxsize, -sys.maxsize - 1, 0
    min_edp, max_edp, sum_edp = sys.maxsize, -sys.maxsize - 1, 0
    min_edp2, max_edp2, sum_edp2 = sys.maxsize, -sys.maxsize - 1, 0

    min_agg_node_hours, max_agg_node_hours, sum_agg_node_hours = sys.maxsize, -sys.maxsize - 1, 0
    # Completion statistics
    throughput = engine.jobs_completed / (engine.current_timestep - engine.timestep_start) * 3600 if \
        (engine.current_timestep - engine.timestep_start != 0) else 0  # Jobs per hour

    min_wait_time, max_wait_time, sum_wait_time = sys.maxsize, -sys.maxsize - 1, 0
    min_turnaround_time, max_turnaround_time, sum_turnaround_time = sys.maxsize, -sys.maxsize - 1, 0
    min_psf_partial_num, max_psf_partial_num, sum_psf_partial_num = sys.maxsize, -sys.maxsize - 1, 0
    min_psf_partial_den, max_psf_partial_den, sum_psf_partial_den = sys.maxsize, -sys.maxsize - 1, 0
    min_awrt, max_awrt, sum_awrt = sys.maxsize, -sys.maxsize - 1, 0

    min_cpu_u, max_cpu_u, sum_cpu_u = sys.maxsize, -sys.maxsize - 1, 0
    min_gpu_u, max_gpu_u, sum_gpu_u = sys.maxsize, -sys.maxsize - 1, 0
    min_ntx_u, max_ntx_u, sum_ntx_u = sys.maxsize, -sys.maxsize - 1, 0
    min_nrx_u, max_nrx_u, sum_nrx_u = sys.maxsize, -sys.maxsize - 1, 0

    jobsSmall = 0
    jobsMedium = 0
    jobsLarge = 0
    jobsVLarge = 0
    jobsHuge = 0

    # Information on Job-Mix
    for job in engine.job_history_dict:
        job_size = job['num_nodes']
        min_job_size, max_job_size, sum_job_size = \
            min_max_sum(job_size, min_job_size, max_job_size, sum_job_size)

        runtime = job['end_time'] - job['start_time']
        min_runtime, max_runtime, sum_runtime = \
            min_max_sum(runtime, min_runtime, max_runtime, sum_runtime)

        energy = job['energy']
        min_energy, max_energy, sum_energy = \
            min_max_sum(energy, min_energy, max_energy, sum_energy)
        edp = energy * runtime
        min_edp, max_edp, sum_edp = \
            min_max_sum(edp, min_edp, max_edp, sum_edp)

        edp2 = energy * runtime**2
        min_edp2, max_edp2, sum_edp2 = \
            min_max_sum(edp2, min_edp2, max_edp2, sum_edp2)

        agg_node_hours = runtime * job_size  # Aggreagte node hours
        min_agg_node_hours, max_agg_node_hours, sum_agg_node_hours = \
            min_max_sum(agg_node_hours, min_agg_node_hours, max_agg_node_hours, sum_agg_node_hours)

        # Completion statistics
        wait_time = job["start_time"] - job["submit_time"]
        min_wait_time, max_wait_time, sum_wait_time = \
            min_max_sum(wait_time, min_wait_time, max_wait_time, sum_wait_time)

        turnaround_time = job["end_time"] - job["submit_time"]
        min_turnaround_time, max_turnaround_time, sum_turnaround_time = \
            min_max_sum(turnaround_time, min_turnaround_time, max_turnaround_time, sum_turnaround_time)

        # Area Weighted Average Response Time
        awrt = agg_node_hours * turnaround_time  # Area Weighted Response Time
        min_awrt, max_awrt, sum_awrt = min_max_sum(awrt, min_awrt, max_awrt, sum_awrt)

        # Priority Weighted Specific Response Time
        psf_partial_num = job_size * (turnaround_time**4 - wait_time**4)
        psf_partial_den = job_size * (turnaround_time**3 - wait_time**3)

        min_psf_partial_num, max_psf_partial_num, sum_psf_partial_num = \
            min_max_sum(psf_partial_num, min_psf_partial_num, max_psf_partial_num, sum_psf_partial_num)
        min_psf_partial_den, max_psf_partial_den, sum_psf_partial_den = \
            min_max_sum(psf_partial_den, min_psf_partial_den, max_psf_partial_den, sum_psf_partial_den)

        if job['avg_cpu_usage'] is not None:
            min_cpu_u, max_cpu_u, sum_cpu_u = min_max_sum(job['avg_cpu_usage'], min_cpu_u, max_cpu_u, sum_cpu_u)
        if job['avg_gpu_usage'] is not None:
            min_gpu_u, max_gpu_u, sum_gpu_u = min_max_sum(job['avg_gpu_usage'], min_gpu_u, max_gpu_u, sum_gpu_u)
        if job['avg_ntx_usage'] is not None:
            min_ntx_u, max_ntx_u, sum_ntx_u = min_max_sum(job['avg_ntx_usage'], min_ntx_u, max_ntx_u, sum_ntx_u)
        if job['avg_nrx_usage'] is not None:
            min_nrx_u, max_nrx_u, sum_nrx_u = min_max_sum(job['avg_nrx_usage'], min_nrx_u, max_nrx_u, sum_nrx_u)

        if job['num_nodes'] <= 5:
            jobsSmall += 1
        elif job['num_nodes'] <= 50:
            jobsMedium += 1
        elif job['num_nodes'] <= 250:
            jobsLarge += 1
        elif job['num_nodes'] <= 4500:
            jobsVLarge += 1
        else:  # job['nodes_required'] > 250:
            jobsHuge += 1

    if len(engine.job_history_dict) != 0:
        avg_job_size = sum_job_size / len(engine.job_history_dict)
        avg_runtime = sum_runtime / len(engine.job_history_dict)
        avg_energy = sum_energy / len(engine.job_history_dict)
        avg_edp = sum_edp / len(engine.job_history_dict)
        avg_edp2 = sum_edp2 / len(engine.job_history_dict)
        avg_agg_node_hours = sum_agg_node_hours / len(engine.job_history_dict)
        avg_wait_time = sum_wait_time / len(engine.job_history_dict)
        avg_turnaround_time = sum_turnaround_time / len(engine.job_history_dict)

        avg_cpu_u = sum_cpu_u / len(engine.job_history_dict)
        avg_gpu_u = sum_gpu_u / len(engine.job_history_dict)
        avg_ntx_u = sum_ntx_u / len(engine.job_history_dict)
        avg_nrx_u = sum_nrx_u / len(engine.job_history_dict)

        if sum_agg_node_hours != 0:
            avg_awrt = sum_awrt / sum_agg_node_hours
        else:
            avg_awrt = 0
        if sum_psf_partial_den != 0:
            psf = (3 * sum_psf_partial_num) / (4 * sum_psf_partial_den)
        else:
            psf = 0
    else:
        # Set these to -1 to indicate nothing ran
        min_job_size, max_job_size, avg_job_size = -1, -1, -1
        min_runtime, max_runtime, avg_runtime = -1, -1, -1
        min_energy, max_energy, avg_energy = -1, -1, -1
        min_edp, max_edp, avg_edp = -1, -1, -1
        min_edp2, max_edp2, avg_edp2 = -1, -1, -1
        min_agg_node_hours, max_agg_node_hours, avg_agg_node_hours = -1, -1, -1
        min_wait_time, max_wait_time, avg_wait_time = -1, -1, -1
        min_turnaround_time, max_turnaround_time, avg_turnaround_time = -1, -1, -1
        min_awrt, max_awrt, avg_awrt = -1, -1, -1
        psf = -1

        min_cpu_u, max_cpu_u, avg_cpu_u = -1, -1, -1
        min_gpu_u, max_gpu_u, avg_gpu_u = -1, -1, -1
        min_ntx_u, max_ntx_u, avg_ntx_u = -1, -1, -1
        min_nrx_u, max_nrx_u, avg_nrx_u = -1, -1, -1

    if min_cpu_u == sys.maxsize and \
       max_cpu_u == -sys.maxsize - 1 and \
       sum_cpu_u == 0:
        min_cpu_u, max_cpu_u, avg_cpu_u = -1, -1, -1

    if min_gpu_u == sys.maxsize and \
       max_gpu_u == -sys.maxsize - 1 and \
       sum_gpu_u == 0:
        min_gpu_u, max_gpu_u, avg_gpu_u = -1, -1, -1
    if min_ntx_u == sys.maxsize and \
       max_ntx_u == -sys.maxsize - 1 and \
       sum_ntx_u == 0:
        min_ntx_u, max_ntx_u, avg_ntx_u = -1, -1, -1

    if min_nrx_u == sys.maxsize and \
       max_nrx_u == -sys.maxsize - 1 and \
       sum_nrx_u == 0:
        min_nrx_u, max_nrx_u, avg_nrx_u = -1, -1, -1

    job_stats = {
        'jobs completed': engine.jobs_completed,
        'throughput': f'{throughput:.2f} jobs/hour',
        'jobs still running': [job.id for job in engine.running],
        'jobs still in queue': [job.id for job in engine.queue],
        'Jobs <= 5 nodes': jobsSmall,
        'Jobs <= 50 nodes': jobsMedium,
        'Jobs <= 250 nodes': jobsLarge,
        'Jobs <= 4500 nodes': jobsVLarge,
        'Jobs > 4500 nodes': jobsHuge,
        # Information on job-mix executed
        'min job size': min_job_size,
        'max job size': max_job_size,
        'average job size': avg_job_size,
        'min runtime': min_runtime,
        'max runtime': max_runtime,
        'average runtime': avg_runtime,
        'min energy': min_energy,
        'max energy': max_energy,
        'avg energy': avg_energy,
        'min edp': min_edp,
        'max edp': max_edp,
        'avg edp': avg_edp,
        'min edp^2': min_edp2,
        'max edp^2': max_edp2,
        'avg edp^2': avg_edp2,
        'min_aggregate_node_hours': min_agg_node_hours,
        'max_aggregate_node_hours': max_agg_node_hours,
        'avg_aggregate_node_hours': avg_agg_node_hours,
        # Utilization:
        'min_cpu_util': min_cpu_u,
        'max_cpu_util': max_cpu_u,
        'avg_cpu_util': avg_cpu_u,
        'min_gpu_util': min_gpu_u,
        'max_gpu_util': max_gpu_u,
        'avg_gpu_util': avg_gpu_u,
        'min_ntx_util': min_ntx_u,
        'max_ntx_util': max_ntx_u,
        'avg_ntx_util': avg_ntx_u,
        'min_nrx_util': min_nrx_u,
        'max_nrx_util': max_nrx_u,
        'avg_nrx_util': avg_nrx_u,
        # Completion statistics
        'min_wait_time': min_wait_time,
        'max_wait_time': max_wait_time,
        'average_wait_time': avg_wait_time,
        'min_turnaround_time': min_turnaround_time,
        'max_turnaround_time': max_turnaround_time,
        'average_turnaround_time': avg_turnaround_time,
        'min_area_weighted_response_time': min_awrt,
        'max_area_weighted_response_time': max_awrt,
        'area_weighted_avg_response_time': avg_awrt,
        'priority_weighted_specific_response_time': psf
    }
    return job_stats


def print_formatted_report(engine_stats=None,
                           job_stats=None,
                           scheduler_stats=None,
                           network_stats=None
                           ):
    # Print a formatted report
    if engine_stats:
        rep_str = "--- Simulation Report ---"
        print(f"\n{rep_str}")
        for key, value in engine_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print(f"{'-' * len(rep_str)}\n")
    if job_stats:
        rep_str = "--- Job Stat Report ---"
        print(f"\n{rep_str}")
        for key, value in job_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print(f"{'-' * len(rep_str)}\n")
    if scheduler_stats:
        rep_str = "--- Scheduler Report ---"
        print(f"\n{rep_str}")
        for key, value in scheduler_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print(f"{'-' * len(rep_str)}\n")
    if network_stats:
        rep_str = "--- Network Report ---"
        print(f"\n{rep_str}")
        for key, value in network_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print(f"{'-' * len(rep_str)}\n")
