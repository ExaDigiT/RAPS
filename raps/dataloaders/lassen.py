"""
Lassen specifications:

    https://hpc.llnl.gov/hardware/compute-platforms/lassen

Reference:

    Patki, Tapasya, et al. "Monitoring large scale supercomputers: A case study with the Lassen supercomputer."
    2021 IEEE International Conference on Cluster Computing (CLUSTER). IEEE, 2021.

Usage Instructions:

    git clone https://github.com/LLNL/LAST/ && cd LAST
    git lfs pull

    # to analyze dataset
    python -m raps.telemetry -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen -v

    # to simulate the dataset as submitted
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen

    # to modify the submit times of the telemetry according to Poisson distribution
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen --arrival poisson

    # to fast-forward 365 days and replay for 1 day. This region day has 2250 jobs with 1650 jobs executed.
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen -ff 365d -t 1d
"""
import math
import os
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from ..job import job_dict
from ..utils import power_to_utilization, next_arrival, convert_to_seconds


def load_data(path, **kwargs):
    """
    Loads data from the given file paths and returns job info.
    """
    nrows = None
    alloc_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_history_hashed.csv'), nrows=nrows, low_memory=False)
    node_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_node_history.csv'), nrows=nrows, low_memory=False)
    step_df = pd.read_csv(os.path.join(path[0], 'final_csm_step_history.csv'), nrows=nrows, low_memory=False)
    return load_data_from_df(alloc_df, node_df, step_df, **kwargs)


def load_data_from_df(allocation_df, node_df, step_df, **kwargs):
    """
    Loads data from pandas DataFrames and returns the extracted job info.
    """
    config = kwargs.get('config')
    jid = kwargs.get('jid', '*')
    validate = kwargs.get('validate')
    arrival = kwargs.get('arrival')
    verbose = kwargs.get('verbose')
    fastforward = kwargs.get('fastforward')  # int in seconds

    allocation_df['job_submit_timestamp'] = pd.to_datetime(allocation_df['job_submit_time'], format='mixed', errors='coerce')
    allocation_df['begin_timestamp'] = pd.to_datetime(allocation_df['begin_time'], format='mixed', errors='coerce')
    allocation_df['end_timestamp'] = pd.to_datetime(allocation_df['end_time'], format='mixed', errors='coerce')

    # Too large dataset! Cut by fastforward and time to simulate!
    if fastforward is None:  # This is in seconds / int?
        fastforward = 0
        fastforward_timedelta = timedelta(seconds=fastforward)  # timedelta
    else:
        fastforward_timedelta = timedelta(seconds=fastforward)  # timedelta
    time_to_simulate = kwargs.get('time')  # int in seconds
    if time_to_simulate is None:  # This is a string!
        time_to_simulate = 31536000  # a year
        time_to_simulate_timedelta = timedelta(seconds=time_to_simulate)  # timedelta
    else:
        time_to_simulate_timedelta = timedelta(seconds=convert_to_seconds(time_to_simulate))  # timedelta

    telemetry_start_timestamp = allocation_df['begin_timestamp'].min()
    telemetry_start_time = 0
    telemetry_end_timestamp = allocation_df['end_timestamp'].max()
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end_time = int(math.ceil(diff.total_seconds()))

    simulation_start_timestamp = telemetry_start_timestamp + fastforward_timedelta
    simulation_end_timestamp = simulation_start_timestamp + time_to_simulate_timedelta

    # As these are >1.4M jobs, filtered to the simulated timestamps before creating the job structs.
    allocation_df = allocation_df[allocation_df['end_timestamp'] >= simulation_start_timestamp]  # Job should not have ended before the simulation time
    allocation_df = allocation_df[allocation_df['job_submit_timestamp'] < simulation_end_timestamp]  # Job has to have been submited before or during the simulaion time

    job_list = []

    for _, row in tqdm(allocation_df.iterrows(), total=len(allocation_df), desc="Processing Jobs"):

        account = row['hashed_user_id']
        job_id = int(row['primary_job_id'])
        allocation_id = row['allocation_id']
        nodes_required = row['num_nodes']
        end_state = row['exit_status']
        name = str(uuid.uuid4())[:6]  # This generates a random 6 char identifier....

        if not jid == '*':
            if int(jid) == int(job_id):
                print(f'Extracting {job_id} profile')
            else:
                continue

        node_data = node_df[node_df['allocation_id'] == row['allocation_id']]

        wall_time = compute_wall_time(row['begin_timestamp'], row['end_timestamp'])
        samples = math.ceil(wall_time / config['TRACE_QUANTA'])

        if validate:
            # Validate should represent the node power and not split it according to cpu and gpu.
            # Not sure if this is correct.
            cpu_power = (node_data['energy'].sum() / nodes_required) / wall_time
            cpu_trace = cpu_power
            gpu_trace = 0  # = cpu_trace  # Is this correct?
        else:
            # Compute GPU power
            gpu_power = (node_data['gpu_energy'].sum() / nodes_required) / wall_time
            gpu_min_power = config['POWER_GPU_IDLE']
            gpu_max_power = config['POWER_GPU_MAX']
            gpu_util = power_to_utilization(gpu_power,gpu_min_power,gpu_max_power)
            gpu_trace = gpu_util

            # Compute CPU power from CPU usage time
            # CPU usage is reported per core, while we need it in the range [0 to CPUS_PER_NODE]
            cpu_util = node_data['cpu_usage'].sum() / nodes_required / wall_time / config['CPU_FREQUENCY'] / config['CORES_PER_CPU']
            cpu_trace = cpu_util
            # TODO use total energy for validation
            # Only Node Energy and GPU Energy is reported!
            # total_energy = node_data['energy'].sum() # Joules

        # Network utilization - since values are given in octets / quarter of a byte, multiply by 4 to get bytes
        ib_tx = 4 * node_data['ib_tx'].sum() if node_data['ib_tx'].values.size > 0 else []
        ib_rx = 4 * node_data['ib_rx'].sum() if node_data['ib_rx'].values.size > 0 else []

        net_tx, net_rx = generate_network_sequences(ib_tx, ib_rx, samples, lambda_poisson=0.3)

        # no priorities defined!
        priority = row.get('priority', 0)
        partition = row.get('partition', "0")

        if arrival == 'poisson':  # Modify the submit times according to Poisson process
            scheduled_nodes = None
            submit_time = next_arrival(1 / config['JOB_ARRIVAL_TIME'])
            start_time = None  # Scheduler will determine start time
            end_time = None  # Scheduler will determine end time
        else:  # Prescribed replay
            scheduled_nodes = get_scheduled_nodes(row['allocation_id'], node_df)
            submit_time = compute_time_offset(row['job_submit_timestamp'], telemetry_start_timestamp)
            start_time = compute_time_offset(row['begin_timestamp'], telemetry_start_timestamp)
            end_time = compute_time_offset(row['end_timestamp'], telemetry_start_timestamp)
            time_limit = row['time_limit']

            trace_time = wall_time
            trace_start_time = start_time
            trace_end_time = end_time
            trace_missing_values = False

        if verbose:
            print('ib_tx, ib_rx, samples:', ib_tx, ib_rx, samples)
            print('tx:', net_tx)
            print('rx:', net_rx)
            print('scheduled_nodes:', nodes_required, scheduled_nodes)

        if wall_time >= 0:
            job_info = job_dict(nodes_required,
                                name,
                                account,
                                cpu_trace,
                                gpu_trace,
                                net_tx,
                                net_rx,
                                end_state,
                                scheduled_nodes,
                                job_id,
                                priority,
                                partition,
                                submit_time=submit_time,
                                time_limit=time_limit,
                                start_time=start_time,
                                end_time=end_time,
                                wall_time=wall_time,
                                trace_time=trace_time,
                                trace_start_time=trace_start_time,
                                trace_end_time=trace_end_time,
                                trace_missing_values=trace_missing_values)

            job_list.append(job_info)

    return job_list, telemetry_start_time, telemetry_end_time


def get_scheduled_nodes(allocation_id, node_df):
    """
    Gets the list of scheduled nodes for a given allocation.
    """
    node_data = node_df[node_df['allocation_id'] == allocation_id]
    if 'node_name' in node_data.columns:
        node_list = [int(node.split('lassen')[-1]) for node in node_data['node_name'].tolist()]
        return node_list
    return []


def compute_wall_time(begin_time, end_time):
    """
    Computes the wall time for the job.
    """
    wall_time = pd.to_datetime(end_time) - pd.to_datetime(begin_time)
    return int(wall_time.total_seconds())


def compute_time_offset(begin_time, reference_time):
    """
    Computes the time offset from a reference time.
    """
    time_offset = pd.to_datetime(begin_time) - reference_time
    return int(time_offset.total_seconds())


def adjust_bursts(burst_intervals, total, intervals):
    bursts = burst_intervals / np.sum(burst_intervals) * total
    bursts = np.round(bursts).astype(int)
    adjustment = total - np.sum(bursts)

    # Distribute adjustment across non-zero elements to avoid negative values
    if adjustment != 0:
        for i in range(len(bursts)):
            if bursts[i] > 0:
                bursts[i] += adjustment % (2^64-1)
                break  # Apply adjustment only once where it won't cause a negative

    return bursts


def generate_network_sequences(total_tx, total_rx, intervals, lambda_poisson):

    if not total_tx or not total_rx:
        return [], []

    # Generate sporadic bursts using a Poisson distribution (shared for both tx and rx)
    burst_intervals = np.random.poisson(lam=lambda_poisson, size=intervals)

    # Ensure some intervals have no traffic (both tx and rx will share zero intervals)
    burst_intervals = np.where(burst_intervals > 0, burst_intervals, 0)

    # Adjust bursts for both tx and rx
    tx_bursts = adjust_bursts(burst_intervals, total_tx, intervals)
    rx_bursts = adjust_bursts(burst_intervals, total_rx, intervals)

    return tx_bursts, rx_bursts


def node_index_to_name(index: int, config: dict):
    """ Converts an index value back to an name string based on system configuration. """
    return f"node{index:04d}"


def cdu_index_to_name(index: int, config: dict):
    return f"cdu{index:02d}"


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    """ Return (row, col) tuple for a cdu index """
    return (0, index) # TODO


if __name__ == "__main__":

    # Example usage
    total_ib_tx = 720  # total transmitted bytes
    total_ib_rx = 480  # total received bytes
    intervals = 20  # number of 20-second intervals
    lambda_poisson = 0.3  # control sporadicity

    tx_sequence, rx_sequence = generate_ib_tx_rx_sequences(total_ib_tx, total_ib_rx, intervals, lambda_poisson)
    print(tx_sequence, rx_sequence)
