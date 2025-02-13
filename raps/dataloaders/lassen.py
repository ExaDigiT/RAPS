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

    # to reschedule
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen --reschedule poisson

    # to fast-forward 37 days and replay for 1 day
    python main.py -f /path/to/LAST/Lassen-Supercomputer-Job-Dataset --system lassen -ff 37d -t 1d
"""
import math
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

try:
    from ..job import job_dict
    from ..utils import power_to_utilization, next_arrival

except:
    pass


def load_data(path, **kwargs):
    """
    Loads data from the given file paths and returns job info.
    """
    nrows = 1E4
    alloc_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_history_hashed.csv'), nrows=nrows)
    node_df = pd.read_csv(os.path.join(path[0], 'final_csm_allocation_node_history.csv'), nrows=nrows)
    step_df = pd.read_csv(os.path.join(path[0], 'final_csm_step_history.csv'), nrows=nrows)
    return load_data_from_df(alloc_df, node_df, step_df, **kwargs)


def load_data_from_df(allocation_df, node_df, step_df, **kwargs):
    """
    Loads data from pandas DataFrames and returns the extracted job info.
    """
    config = kwargs.get('config')
    jid = kwargs.get('jid', '*')
    reschedule = kwargs.get('reschedule')
    fastforward = kwargs.get('fastforward')
    verbose = kwargs.get('verbose')
    min_time = kwargs.get('min_time', None)

    if fastforward:
        print(f"fast-forwarding {fastforward} seconds")

    allocation_df['begin_time'] = pd.to_datetime(allocation_df['begin_time'], format='mixed', errors='coerce')
    allocation_df['end_time'] = pd.to_datetime(allocation_df['end_time'], format='mixed', errors='coerce')

    if not min_time:
        min_time = pd.to_datetime(allocation_df['begin_time']).min()

    job_list = []

    for _, row in tqdm(allocation_df.iterrows(), total=len(allocation_df), desc="Processing Jobs"):
        job_id = row['primary_job_id']

        if not jid == '*':
            if int(jid) == int(job_id):
                print(f'Extracting {job_id} profile')
            else:
                continue

        node_data = node_df[node_df['allocation_id'] == row['allocation_id']]

        nodes_required = row['num_nodes']

        wall_time = compute_wall_time(row['begin_time'], row['end_time'])
        samples = math.ceil(wall_time / config['TRACE_QUANTA'])

        # Compute GPU power
        gpu_energy = node_data['gpu_energy'].sum()  # Joules
        # divide by nodes_required to get average gpu_usage per node
        gpu_usage = node_data['gpu_usage'].sum() / 1E6 / nodes_required  # seconds
        gpu_power = gpu_energy / gpu_usage if gpu_usage > 0 else 0
        #gpu_power = gpu_energy / wall_time
        gpu_power_array = np.array([gpu_power] * samples)

        gpu_min_power = nodes_required * config['POWER_GPU_IDLE']
        gpu_max_power = nodes_required * config['POWER_GPU_MAX']
        gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
        # GPU power can be 0:
        # Utilization is defined in the range of [0 to GPUS_PER_NODE].
        # gpu_util will be negative if power reports 0, which is smaller than POWER_GPU_IDLE
        # Therefore: gpu_util should be set to zero if it is smaller than 0.
        gpu_trace = np.maximum(0, gpu_util)

        # Compute CPU power from CPU usage time
        # CPU usage is reported per core, while we need it in the range [0 to CPUS_PER_NODE]
        cpu_usage = node_data['cpu_usage'].sum() / 1E9 / nodes_required / config['CORES_PER_CPU'] # seconds
        cpu_usage_array = np.array([cpu_usage] * samples)
        cpu_util = cpu_usage_array / wall_time
        cpu_trace = cpu_util  # * CPUS_PER_NODE
        # TODO use total energy for validation
        # Only Node Energy and GPU Energy is reported!
        # total_energy = node_data['energy'].sum() # Joules

        # Network utilization - since values are given in octets / quarter of a byte, multiply by 4 to get bytes
        ib_tx = 4 * node_data['ib_tx'].values[0] if node_data['ib_tx'].values.size > 0 else []
        ib_rx = 4 * node_data['ib_rx'].values[0] if node_data['ib_rx'].values.size > 0 else []

        net_tx, net_rx = generate_network_sequences(ib_tx, ib_rx, samples, lambda_poisson=0.3)

        if reschedule == 'poisson':  # Let the scheduler reschedule the jobs
            scheduled_nodes = None
            time_offset = next_arrival(1/config['JOB_ARRIVAL_TIME'])
        else:
            scheduled_nodes = get_scheduled_nodes(row['allocation_id'], node_df)
            time_offset = compute_time_offset(row['begin_time'], min_time)
            if fastforward:
                time_offset -= fastforward

        if verbose:
            print('ib_tx, ib_rx, samples:', ib_tx, ib_rx, samples)
            print('tx:', net_tx)
            print('rx:', net_rx)
            print('scheduled_nodes:', nodes_required, scheduled_nodes)

        if time_offset >= 0:

            job_info = job_dict(nodes_required,
                                row['hashed_user_id'],
                                row['hashed_user_group_id'],
                                cpu_trace, gpu_trace, net_tx, net_rx, wall_time,
                                row['exit_status'],
                                scheduled_nodes,
                                time_offset,
                                job_id,
                                row.get('priority', 0))

            job_list.append(job_info)

    return job_list


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
                bursts[i] += adjustment
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
