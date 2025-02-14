"""
This module provides functionality for generating statistics.
These are statistics on
the engine
the jobs

Both could be part of the engine or jobs class, but as the are very verbose, try to keep statistics consolidated in this file.
"""
import sys
from .utils import sum_values, min_value, max_value

from .engine import Engine


def get_engine_stats(engine: Engine):
    """ Return engine statistics """
    num_samples = len(engine.power_manager.history) if engine.power_manager else 0

    average_power_mw = sum_values(engine.power_manager.history) / num_samples / 1000 if num_samples else 0
    average_loss_mw = sum_values(engine.power_manager.loss_history) / num_samples / 1000 if num_samples else 0
    min_loss_mw = min_value(engine.power_manager.loss_history) / 1000 if num_samples else 0
    max_loss_mw = max_value(engine.power_manager.loss_history) / 1000 if num_samples else 0

    loss_fraction = average_loss_mw / average_power_mw if average_power_mw else 0
    efficiency = 1 - loss_fraction if loss_fraction else 0
    total_energy_consumed = average_power_mw * engine.timesteps / 3600 if engine.timesteps else 0  # MW-hr
    emissions = total_energy_consumed * 852.3 / 2204.6 / efficiency if efficiency else 0
    total_cost = total_energy_consumed * 1000 * engine.config.get('POWER_COST', 0)  # Total cost in dollars

    stats = {
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

    return stats


def min_max_sum(value,min,max,sum):
    if value < min:
        min = value
    if value > max:
        max = value
    sum += value
    return min,max,sum


def get_job_stats(engine: Engine):
    """ Return job statistics processed over the engine execution"""
    # Information on Job-Mix
    min_job_size, max_job_size, sum_job_size = sys.maxsize, -sys.maxsize - 1, 0
    min_runtime, max_runtime, sum_runtime = sys.maxsize, -sys.maxsize - 1, 0
    min_agg_node_hours, max_agg_node_hours, sum_agg_node_hours = sys.maxsize, -sys.maxsize - 1, 0
    # Completion statistics
    throughput = engine.jobs_completed / engine.timesteps * 3600 if engine.timesteps else 0  # Jobs per hour

    min_wait_time, max_wait_time, sum_wait_time = sys.maxsize, -sys.maxsize - 1, 0
    min_turnaround_time, max_turnaround_time, sum_turnaround_time = sys.maxsize, -sys.maxsize - 1, 0

    min_awrt, max_awrt, sum_awrt = sys.maxsize, -sys.maxsize - 1, 0

    # Information on Job-Mix
    for job in engine.job_history_dict:
        job_size = job['num_nodes']
        min_job_size,max_job_size,sum_job_size = \
            min_max_sum(job_size, min_job_size, max_job_size, sum_job_size)

        runtime = job['end_time'] - job['start_time']
        min_runtime, max_runtime, sum_runtime = \
            min_max_sum(runtime, min_runtime, max_runtime, sum_runtime)

        agg_node_hours = runtime * job_size  # Aggreagte node hours
        min_agg_node_hours, max_agg_node_hours, sum_agg_node_hours = \
            min_max_sum(agg_node_hours, min_agg_node_hours, max_agg_node_hours, sum_agg_node_hours)

        # Completion statistics
        wait_time = job["start_time"] - job["submit_time"]
        min_wait_time,max_wait_time,sum_wait_time = \
            min_max_sum(wait_time, min_wait_time, max_wait_time, sum_wait_time)

        turnaround_time = job["end_time"] - job["submit_time"]
        min_turnaround_time, max_turnaround_time, sum_turnaround_time = \
            min_max_sum(turnaround_time, min_turnaround_time, max_turnaround_time, sum_turnaround_time)

        awrt = agg_node_hours * turnaround_time  # Area Weighted Response Time
        min_awrt, max_awrt, sum_awrt = min_max_sum(awrt, min_awrt, max_awrt, sum_awrt)

    if len(engine.job_history_dict) != 0:
        avg_job_size = sum_job_size / len(engine.job_history_dict)
        avg_runtime = sum_runtime / len(engine.job_history_dict)
        avg_agg_node_hours = sum_agg_node_hours / len(engine.job_history_dict)
        avg_wait_time = sum_wait_time / len(engine.job_history_dict)
        avg_turnaround_time = sum_turnaround_time / len(engine.job_history_dict)
        avg_awrt = sum_awrt / len(engine.job_history_dict)
    else:
        # Set these to -1 to indicate nothing ran
        min_job_size, max_job_size, avg_job_size = -1,-1,-1
        min_runtime, max_runtime, avg_runtime = -1,-1,-1
        min_agg_node_hours, max_agg_node_hours, avg_agg_node_hours = -1,-1,-1
        min_wait_time, max_wait_time, avg_wait_time = -1,-1,-1
        min_turnaround_time, max_turnaround_time, avg_turnaround_time = -1,-1,-1
        min_awrt, max_awrt, avg_awrt = -1,-1,-1

    job_stats = {
        'jobs completed': engine.jobs_completed,
        'throughput': f'{throughput:.2f} jobs/hour',
        'jobs still running': [job.id for job in engine.running],
        'jobs still in queue': [job.id for job in engine.queue],
        # Information on job-mix executed
        'min job size': min_job_size,
        'max job size': max_job_size,
        'average job size': avg_job_size,
        'min runtime': min_runtime,
        'max runtime': max_runtime,
        'average runtime': avg_runtime,
        'min_aggregate_node_hours': min_agg_node_hours,
        'max_aggregate_node_hours': max_agg_node_hours,
        'avg_aggregate_node_hours': avg_agg_node_hours,
        # Completion statistics
        'min_wait_time': min_wait_time,
        'max_wait_time': max_wait_time,
        'average_wait_time': avg_wait_time,
        'min_turnaround_time': min_turnaround_time,
        'max_turnaround_time': max_turnaround_time,
        'average_turnaround_time': avg_turnaround_time,
        'min_area_weighted_response_time': min_awrt,
        'max_area_weighted_response_time': max_awrt,
        'avg_area_weighted_response_time': avg_awrt
    }
    return job_stats
