"""
    # Reference
    Antici, Francesco, et al. "PM100: A Job Power Consumption Dataset of a
    Large-scale Production HPC System." Proceedings of the SC'23 Workshops
    of The International Conference on High Performance Computing,
    Network, Storage, and Analysis. 2023.

    # get the data
    Download `job_table.parquet` from https://zenodo.org/records/10127767

    # to simulate the dataset
    python main.py -f /path/to/job_table.parquet --system marconi100

    # to reschedule
    python main.py -f /path/to/job_table.parquet --system marconi100 --reschedule poisson

    # to fast-forward 60 days and replay for 1 day
    python main.py -f /path/to/job_table.parquet --system marconi100 -ff 60d -t 1d

    # to analyze dataset
    python -m raps.telemetry -f /path/to/job_table.parquet --system marconi100 -v

"""
import uuid
import random
import pandas as pd
from tqdm import tqdm

from ..job import job_dict
from ..utils import power_to_utilization, next_arrival


def load_data(jobs_path, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Parameters
    ----------
    jobs_path : str
        The path to the jobs parquet file.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    jobs_df = pd.read_parquet(jobs_path, engine='pyarrow')
    return load_data_from_df(jobs_df, **kwargs)


def load_data_from_df(jobs_df: pd.DataFrame, **kwargs):
    """
    Reads job and job profile data from parquet files and parses them.

    Returns
    -------
    list
        The list of parsed jobs.
    """
    config = kwargs.get('config')
    min_time = kwargs.get('min_time', None)
    reschedule = kwargs.get('reschedule')
    fastforward = kwargs.get('fastforward')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')

    if fastforward: print(f"fast-forwarding {fastforward} seconds")

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df.sort_values(by='start_time')
    jobs_df = jobs_df.reset_index(drop=True)

    # Take earliest time as baseline reference
    # We can use the start time of the first job.
    if min_time:
        time_zero = min_time
    else:
        time_zero = jobs_df['start_time'].min()

    num_jobs = len(jobs_df)
    print("time_zero:", time_zero, "num_jobs", num_jobs)

    jobs = []

    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Jobs"):

        account = jobs_df.loc[jidx, 'user_id'] # or 'group_id'
        job_id = jobs_df.loc[jidx, 'job_id']

        if not jid == '*':
            if int(jid) == int(job_id):
                print(f'Extracting {job_id} profile')
            else:
                continue
        nodes_required = jobs_df.loc[jidx, 'num_nodes_alloc']

        name = str(uuid.uuid4())[:6]

        if validate:
            cpu_power = jobs_df.loc[jidx, 'node_power_consumption']/jobs_df.loc[jidx, 'num_nodes_alloc']
            cpu_trace = cpu_power
            gpu_trace = cpu_trace

        else:
            cpu_power = jobs_df.loc[jidx, 'cpu_power_consumption']
            cpu_power_array = cpu_power.tolist()
            cpu_min_power = nodes_required * config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
            cpu_max_power = nodes_required * config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            cpu_trace = cpu_util * config['CPUS_PER_NODE']

            node_power = (jobs_df.loc[jidx, 'node_power_consumption']).tolist()
            mem_power = (jobs_df.loc[jidx, 'mem_power_consumption']).tolist()
            # Find the minimum length among the three lists
            min_length = min(len(node_power), len(cpu_power), len(mem_power))
            # Slice each list to the minimum length
            node_power = node_power[:min_length]
            cpu_power = cpu_power[:min_length]
            mem_power = mem_power[:min_length]

            gpu_power = (node_power - cpu_power - mem_power
                - ([nodes_required * config['NICS_PER_NODE'] * config['POWER_NIC']] * len(node_power))
                - ([nodes_required * config['POWER_NVME']] * len(node_power)))
            gpu_power_array = gpu_power.tolist()
            gpu_min_power = nodes_required * config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']
            gpu_max_power = nodes_required * config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']
            gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            gpu_trace = gpu_util * config['GPUS_PER_NODE']

        priority = int(jobs_df.loc[jidx, 'priority'])

        wall_time = gpu_trace.size * config['TRACE_QUANTA'] # seconds
        end_state = jobs_df.loc[jidx, 'job_state']

        time_start = jobs_df.loc[jidx+1, 'start_time']
        time_start = time_start - time_zero

        time_submit = jobs_df.loc[jidx, 'submit_time']
        time_submit = time_submit - time_zero

        if jid == '*':
            time_submit = max(time_submit.total_seconds(), 0)
        else:
            # When extracting out a single job, run one iteration past the end of the job
            time_submit = config['UI_UPDATE_FREQ']

        if fastforward:
            time_start -= fastforward
            time_submit -= fastforward

        if reschedule == 'poisson':  # Let the scheduler reschedule the jobs
            scheduled_nodes = None
            time_submit = next_arrival(1/config['JOB_ARRIVAL_TIME'])
            time_start = None
        elif reschedule == 'submit-time':
            raise NotImplementedError
        else:  # Prescribed replay
            scheduled_nodes = (jobs_df.loc[jidx, 'nodes']).tolist()

        if gpu_trace.size > 0 and time_submit >= 0:
            job_info = job_dict(nodes_required, name, account, cpu_trace, gpu_trace, [], [], wall_time,
                                end_state, scheduled_nodes, time_submit, job_id, priority, time_start)
            jobs.append(job_info)

    return jobs


def node_index_to_name(index: int, config: dict):
    """ Converts an index value back to an name string based on system configuration. """
    return f"node{index:04d}"


def cdu_index_to_name(index: int, config: dict):
    return f"cdu{index:02d}"


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    """ Return (row, col) tuple for a cdu index """
    return (0, index) # TODO
