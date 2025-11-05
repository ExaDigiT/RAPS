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
    """
    Return engine statistics
    """
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
        'time_simulated': time_simulated,
        'num_samples': num_samples,
        'average_power': average_power_mw,
        'min_loss': min_loss_mw,
        'average_loss': average_loss_mw,
        'max_loss': max_loss_mw,
        'system_power_efficiency': efficiency * 100,
        'total_energy_consumed': total_energy_consumed,
        'carbon_emissions': emissions,
        'total_cost': total_cost,
    }

    if engine.config['multitenant']:
        # Multitenancy Stats
        total_jobs_loaded = engine.total_initial_jobs  # Assuming this is passed to __init__
        stats['total_jobs_loaded'] = total_jobs_loaded
        if total_jobs_loaded > 0:
            stats['jobs_completed_percentage'] = engine.jobs_completed / total_jobs_loaded * 100
        else:
            stats['jobs_completed_percentage'] = 0

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

            stats['avg_concurrent_jobs_per_active_node'] = avg_jobs_per_active_node
            stats['max_concurrent_jobs_per_node'] = max_concurrent_jobs_per_node
    else:
        stats['avg_concurrent_jobs_per_node'] = None
        stats['max_concurrent_jobs_per_node'] = None

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

    stats["avg_network_util"] = mean_net_util * 100

    if engine.avg_slowdown_history:
        avg_job_slow = sum(engine.avg_slowdown_history) / len(engine.avg_slowdown_history)
    else:
        avg_job_slow = 1.0
    stats["avg_per_job_slowdown"] = avg_job_slow

    if engine.max_slowdown_history:
        max_job_slow = max(engine.max_slowdown_history)
    else:
        max_job_slow = 1.0
    stats["max_per_job_slowdown"] = max_job_slow

    if engine.net_congestion_history:
        congestion_values = [c for t, c in engine.net_congestion_history]
        stats['avg_inter_job_congestion'] = sum(congestion_values) / len(congestion_values)
        stats['max_inter_job_congestion'] = max(congestion_values)
        stats['min_inter_job_congestion'] = min(congestion_values)
    else:
        stats['avg_inter_job_congestion'] = 0.0
        stats['max_inter_job_congestion'] = 0.0
        stats['min_inter_job_congestion'] = 0.0

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
        'jobs_total': engine.jobs_completed + len(engine.running) + len(engine.queue),
        'jobs_completed': engine.jobs_completed,
        'throughput': throughput,
        'jobs_still_running': [job.id for job in engine.running],
        'jobs_still_in_queue': [job.id for job in engine.queue],
        'jobs <= 5 nodes': jobsSmall,
        'jobs <= 50 nodes': jobsMedium,
        'jobs <= 250 nodes': jobsLarge,
        'jobs <= 4500 nodes': jobsVLarge,
        'jobs > 4500 nodes': jobsHuge,
        # Information on job-mix executed
        'min_job_size': min_job_size,
        'max_job_size': max_job_size,
        'average_job_size': avg_job_size,
        'min_runtime': min_runtime,
        'max_runtime': max_runtime,
        'average_runtime': avg_runtime,
        'min_energy': min_energy,
        'max_energy': max_energy,
        'avg_energy': avg_energy,
        'min_edp': min_edp,
        'max_edp': max_edp,
        'avg_edp': avg_edp,
        'min_edp^2': min_edp2,
        'max_edp^2': max_edp2,
        'avg_edp^2': avg_edp2,
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


def get_stats(engine: Engine):
    return {
        'engine': get_engine_stats(engine),
        'job': get_job_stats(engine),
        'scheduler': get_scheduler_stats(engine),
        'network': get_network_stats(engine) if engine.simulate_network else {},
    }


def print_formatted_report(engine_stats=None,
                           job_stats=None,
                           scheduler_stats=None,
                           network_stats=None
                           ):
    def print_report_section(name, data, templates):
        if data:
            rep_str = f"--- {name} ---"
            print(rep_str)
            for key, value in data.items():
                pretty_key = key.replace('_', ' ').title()
                if key in templates:
                    pretty_value = templates[key].format(value)
                elif isinstance(value, float):
                    pretty_value = f"{value:.2f}"
                elif value is None:
                    pretty_value = "N/A"
                else:
                    pretty_value = str(value)
                print(f"{pretty_key}: {pretty_value}")
            print(f"{'-' * len(rep_str)}\n")
            print()

    # Print a formatted report
    print()
    print_report_section("Simulation Report", engine_stats, {
        'average_power': '{:.4f} MW',
        'min_loss': '{:.4f} MW',
        'average_loss': '{:.2f} MW',
        'max_loss': '{:.2f} MW',
        'system_power_efficiency': '{:.2f}%',
        'total_energy_consumed': '{:.2f} MW-hr',
        'carbon_emissions': '{:.4f} metric tons CO2',
        'total_cost': '${:.2f}',
    })
    print_report_section("Job Stat Report", job_stats, {
        'throughput': '{:.2f} jobs/hour',
        'jobs_completed_percentage': "{:.2f}%",
    })
    print_report_section("Scheduler Report", scheduler_stats, {
    })
    print_report_section("Network Report", network_stats, {
        "avg_network_util": "{:.2f}%",
        "avg_per_job_slowdown": "{:.2f}x",
        "max_per_job_slowdown": "{:.2f}x",
        "avg_inter_job_congestion": "{:.2f}",
        "max_inter_job_congestion": "{:.2f}",
        "min_inter_job_congestion": "{:.2f}",
    })


def get_gauge_limits(engine: Engine):
    """For setting max values in dashboard gauges"""
    peak_flops = engine.flops_manager.get_rpeak()
    peak_power = engine.power_manager.get_peak_power()
    gflops_per_watt_max = peak_flops / 1E9 / peak_power

    return {
        'peak_flops': peak_flops,
        'peak_power': peak_power,
        'g_flops_w_peak': gflops_per_watt_max
    }


class RunningStats:
    """
    Calculate a subset of the stats in as "running totals" for each engine tick. This is much more
    efficient than calling get_engine_stats() repeatedly.
    """
    # TODO: maybe should combine this and get_engine_stats logic?
    @staticmethod
    def _running_stats(engine: Engine):
        # Infinite generator used for the RunningStats logic
        def running_sum_values(values, last_value, last_index):
            return last_value + sum_values(values[last_index:])

        def running_min_value(values, last_value, last_index):
            if last_index < len(values):
                new_min = min_value(values[last_index:])
                rtrn = new_min if last_value is None else min(new_min, last_value)
            else:
                rtrn = last_value  # No change
            return rtrn

        def running_max_value(values, last_value, last_index):
            if last_index < len(values):
                new_max = max_value(values[last_index:])
                return new_max if last_value is None else max(new_max, last_value)
            else:
                return last_value  # No change

        last_power_index = 0
        power_sum = 0
        last_loss_index = 0
        loss_sum = 0
        loss_min = None
        loss_max = None

        while True:
            timesteps = engine.current_timestep - engine.timestep_start
            throughput = engine.jobs_completed / timesteps * 3600 if timesteps != 0 else 0  # Jobs per hour
            num_samples = len(engine.power_manager.history) if engine.power_manager else 0

            power_sum = running_sum_values(engine.power_manager.history, power_sum, last_power_index)
            average_power_mw = power_sum / num_samples / 1000 if num_samples else 0
            last_power_index = len(engine.power_manager.history)

            loss_sum = running_sum_values(engine.power_manager.loss_history, loss_sum, last_loss_index)
            average_loss_mw = loss_sum / num_samples / 1000 if num_samples else 0
            loss_min = running_min_value(engine.power_manager.loss_history, loss_min, last_loss_index)
            min_loss_mw = loss_min / 1000 if num_samples else 0
            loss_max = running_max_value(engine.power_manager.loss_history, loss_max, last_loss_index)
            max_loss_mw = loss_max / 1000 if num_samples else 0
            last_loss_index = len(engine.power_manager.loss_history)

            loss_fraction = average_loss_mw / average_power_mw if average_power_mw else 0
            efficiency = 1 - loss_fraction if loss_fraction else 0
            total_energy_consumed = average_power_mw * timesteps / 3600 if timesteps else 0  # MW-hr
            carbon_emissions = total_energy_consumed * 852.3 / 2204.6 / efficiency if efficiency else 0
            total_cost = total_energy_consumed * 1000 * engine.config.get('POWER_COST', 0)  # Total cost in dollars

            yield {
                "throughput": throughput,
                "num_samples": num_samples,
                "average_power": average_power_mw,
                "min_loss": min_loss_mw,
                "average_loss": average_loss_mw,
                "max_loss": max_loss_mw,
                "system_power_efficiency": efficiency * 100,
                "total_energy_consumed": total_energy_consumed,
                "carbon_emissions": carbon_emissions,
                "total_cost": total_cost,
            }

    def __init__(self, engine: Engine):
        self._gen = RunningStats._running_stats(engine)

    def get_stats(self) -> dict:
        return next(self._gen)
