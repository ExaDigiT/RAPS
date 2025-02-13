"""
    Note: Frontier telemetry data is not publicly available.

    # To simulate
    DATEDIR="date=2024-01-18"
    DPATH=/path/to/data
    python main.py -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/$DATEDIR

    # To analyze the data
    python -m raps.telemetry -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/$DATEDIR
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..job import job_dict
from ..utils import power_to_utilization, next_arrival, encrypt


def aging_boost(nnodes):
    """Frontier aging policy as per documentation here:
       https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#job-priority-by-node-count
    """
    if nnodes >= 5645:
        return 8*24*3600  # seconds
    elif nnodes >= 1882:
        return 4*24*3600
    else:
        return 0


def load_data(files, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    assert (len(files) == 2), "Frontier dataloader requires two files: joblive and jobprofile"

    jobs_path = files[0]
    jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')

    jobprofile_path = files[1]
    jobprofile_df = pd.read_parquet(jobprofile_path, engine='pyarrow')

    return load_data_from_df(jobs_df, jobprofile_df, **kwargs)


def load_data_from_df(jobs_df: pd.DataFrame, jobprofile_df: pd.DataFrame, **kwargs):
    """
    Reads job and job profile data from dataframes files and parses them.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    config = kwargs.get('config')
    encrypt_bool = kwargs.get('encrypt')
    fastforward = kwargs.get('fastforward')
    reschedule = kwargs.get('reschedule')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')

    if fastforward:
        print(f"fast-forwarding {fastforward} seconds")

    min_time = kwargs.get('min_time', None)

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df[jobs_df['time_start'].notna()]
    jobs_df = jobs_df.drop_duplicates(subset='job_id', keep='last').reset_index()
    jobs_df = jobs_df.sort_values(by='time_start')
    jobs_df = jobs_df.reset_index(drop=True)

    # Convert timestamp column to datetime format
    jobprofile_df['timestamp'] = pd.to_datetime(jobprofile_df['timestamp'])

    # Sort allocation dataframe based on timestamp, adjust indices after sorting
    jobprofile_df = jobprofile_df.sort_values(by='timestamp')
    jobprofile_df = jobprofile_df.reset_index(drop=True)

    # Take earliest time as baseline reference
    if min_time:
        time_zero = min_time
    else:
        time_zero = jobs_df['time_snapshot'].min()

    num_jobs = len(jobs_df)
    print("time_zero:", time_zero, "num_jobs", num_jobs)

    jobs = []
    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Jobs"):

        user = jobs_df.loc[jidx, 'user']
        account = jobs_df.loc[jidx, 'account']
        job_id = jobs_df.loc[jidx, 'job_id']
        allocation_id = jobs_df.loc[jidx, 'allocation_id']
        nodes_required = jobs_df.loc[jidx, 'node_count']
        end_state = jobs_df.loc[jidx, 'state_current']
        name = jobs_df.loc[jidx, 'name']
        if encrypt_bool:
            name = encrypt(name)

        if validate:
            cpu_power = jobprofile_df[jobprofile_df['allocation_id']
                                      == allocation_id]['mean_node_power']
            cpu_trace = cpu_power.values
            gpu_trace = cpu_trace

        else:
            cpu_power = jobprofile_df[jobprofile_df['allocation_id']
                                      == allocation_id]['sum_cpu0_power']
            cpu_power_array = cpu_power.values
            cpu_min_power = nodes_required * config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
            cpu_max_power = nodes_required * config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            cpu_trace = cpu_util * config['CPUS_PER_NODE']

            gpu_power = jobprofile_df[jobprofile_df['allocation_id']
                                      == allocation_id]['sum_gpu_power']
            gpu_power_array = gpu_power.values

            gpu_min_power = nodes_required * config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']
            gpu_max_power = nodes_required * config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']
            gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            gpu_trace = gpu_util * config['GPUS_PER_NODE']

        # Set any NaN values in cpu_trace and/or gpu_trace to zero
        cpu_trace[np.isnan(cpu_trace)] = 0
        gpu_trace[np.isnan(gpu_trace)] = 0

        wall_time = gpu_trace.size * config['TRACE_QUANTA']  # seconds

        time_submit = jobs_df.loc[jidx, 'time_submission']
        diff = time_submit - time_zero
        time_submit = max(diff.total_seconds(), 0)

        time_start = jobs_df.loc[jidx, 'time_start']
        diff = time_start - time_zero
        time_start = max(diff.total_seconds(), 0)

        if fastforward:
            time_start -= fastforward
            time_submit -= fastforward

        xnames = jobs_df.loc[jidx, 'xnames']
        # Don't replay any job with an empty set of xnames
        if '' in xnames:
            continue

        if reschedule == 'poisson':  # Let the scheduler reschedule the jobs
            scheduled_nodes = None
            time_offset = next_arrival(1/config['JOB_ARRIVAL_TIME'])
            time_start = None
            priority = aging_boost(nodes_required)

        else:  # Prescribed replay
            scheduled_nodes = []
            priority = 0  # not used for replay
            for xname in xnames:
                indices = xname_to_index(xname, config)
                scheduled_nodes.append(indices)

        if gpu_trace.size == 0:
            print("ignoring job b/c zero trace:", jidx, time_submit, time_start, nodes_required)

        if gpu_trace.size > 0 and (jid == job_id or jid == '*') and time_submit >= 0:
            job_info = job_dict(nodes_required, name, account, cpu_trace, gpu_trace, [], [], wall_time,
                                end_state, scheduled_nodes, time_submit, job_id, priority, time_start)
            jobs.append(job_info)

    return jobs


def xname_to_index(xname: str, config: dict):
    """
    Converts an xname string to an index value based on system configuration.

    Parameters
    ----------
    xname : str
        The xname string to convert.

    Returns
    -------
    int
        The index value corresponding to the xname.
    """
    row, col = int(xname[2]), int(xname[3:5])
    chassis, slot, node = int(xname[6]), int(xname[8]), int(xname[10])
    if row == 6:
        col -= 9
    rack_index = row * 12 + col
    node_index = chassis * config['BLADES_PER_CHASSIS'] * config['NODES_PER_BLADE'] + slot * config['NODES_PER_BLADE'] + node
    return rack_index * config['SC_SHAPE'][2] + node_index


def node_index_to_name(index: int, config: dict):
    """
    Converts an index value back to an xname string based on system configuration.

    Parameters
    ----------
    index : int
        The index value to convert.

    Returns
    -------
    str
        The xname string corresponding to the index.
    """
    rack_index = index // config['SC_SHAPE'][2]
    node_index = index % config['SC_SHAPE'][2]

    row = rack_index // 12
    col = rack_index % 12
    if row == 6:
        col += 9

    chassis = node_index // (config['BLADES_PER_CHASSIS'] * config['NODES_PER_BLADE'])
    remaining = node_index % (config['BLADES_PER_CHASSIS'] * config['NODES_PER_BLADE'])
    slot = remaining // config['NODES_PER_BLADE']
    node = remaining % config['NODES_PER_BLADE']

    return f"x2{row}{col:02}c{chassis}s{slot}b{node}"


CDU_NAMES = [
    'x2002c1', 'x2003c1', 'x2006c1', 'x2009c1', 'x2102c1', 'x2103c1', 'x2106c1', 'x2109c1',
    'x2202c1', 'x2203c1', 'x2206c1', 'x2209c1', 'x2302c1', 'x2303c1', 'x2306c1', 'x2309c1',
    'x2402c1', 'x2403c1', 'x2406c1', 'x2409c1', 'x2502c1', 'x2503c1', 'x2506c1', 'x2509c1',
    'x2609c1',
]


def cdu_index_to_name(index: int, config: dict):
    return CDU_NAMES[index - 1]


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    """ Return (row, col) tuple for a cdu index """
    name = CDU_NAMES[index - 1]
    row, col = int(name[2]), int(name[3:5])
    return (row, col)
