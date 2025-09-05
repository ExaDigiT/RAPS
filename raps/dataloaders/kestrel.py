"""
    Load data for NREL's Kestrel cluster.
"""
import uuid
import pandas as pd
from tqdm import tqdm

from ..job import job_dict, Job
from ..utils import power_to_utilization, next_arrival, WorkloadData


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

    Requires the following fields in the DataFrame:
    - start_time (timestamp): Time execution begins (actual or expected) 	
    - job_id (int): Job ID
    - node_power_consumption (List[int]): Power consumption of the job, recorded at Node level
    - nodes_required (int): Number of nodes allocated to the job
    - cpu_power_consumption (List[int]): Power consumption of the job, recorded at CPU level (don't have this)
    - mem_power_consumption (List[int]): Power consumption of the job, recorded at Memory level (don't have this)
    - priority (int): Relative priority of the job, 0=held, 1=required nodes DOWN/DRAINED
    - job_state (string): State of the job, see enum job_states for possible values
    - wall_time (int): Actual runtime of job, in seconds
    - nodes (string): List of nodes allocated to job

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
    jobs_df = jobs_df.sort_values(by='submit_time')
    jobs_df = jobs_df[(jobs_df.start_time.between(pd.to_datetime('2024-09-01T00:00:00'), 
                                                 pd.to_datetime('2024-09-16T00:00:00'), inclusive='right') | 
                        jobs_df.end_time.between(pd.to_datetime('2024-09-01T00:00:00'), 
                                                 pd.to_datetime('2024-09-16T00:00:00'), inclusive='right') 
                                                 )].copy()
    jobs_df = jobs_df.reset_index(drop=True)

    telemetry_start_timestamp = jobs_df['start_time'].min()
    telemetry_end_timestamp = jobs_df['end_time'].max()
    telemetry_start = 0
    telemetry_end = int((telemetry_end_timestamp - telemetry_start_timestamp).total_seconds())

    # Take earliest time as baseline reference
    # We can use the start time of the first job.
    if min_time:
        time_zero = min_time
    else:
        time_zero = jobs_df['submit_time'].min()

    num_jobs = len(jobs_df)
    print("time_zero:", time_zero, "num_jobs", num_jobs)

    jobs = []

    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Kestrel Jobs"):

        job_id = jobs_df.loc[jidx, 'job_id']
        account = jobs_df.loc[jidx, 'account']

        if not jid == '*': 
            if int(jid) == int(job_id): 
                print(f'Extracting {job_id} profile')
            else:
                continue
        nodes_required = jobs_df.loc[jidx, 'nodes_required']

        name = str(uuid.uuid4())[:6]
            
        if validate:
            cpu_power = jobs_df.loc[jidx, 'power_per_node']
            cpu_trace = cpu_power

        else:                
            cpu_power = jobs_df.loc[jidx, 'power_per_node']
            cpu_power_array = [600] if (pd.isna(cpu_power) or cpu_power == 0) else cpu_power.tolist()
            cpu_min_power = nodes_required * config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
            cpu_max_power = nodes_required * config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            cpu_trace = cpu_util * config['CPUS_PER_NODE']
            gpu_trace = 0
                
        # Priority sorting doesn't seem to be implemented at the moment
        priority = 0
            
        wall_time = jobs_df.loc[jidx, 'wall_time']
        end_state = jobs_df.loc[jidx, 'job_state']
        time_submit = jobs_df.loc[jidx+1, 'submit_time']
        diff = time_submit - time_zero

        if jid == '*': 
            time_offset = max(diff.total_seconds(), 0)
        else:
            # When extracting out a single job, run one iteration past the end of the job
            time_offset = config['UI_UPDATE_FREQ']

        if fastforward: time_offset -= fastforward

        if reschedule: # Let the scheduler reschedule the jobs
            scheduled_nodes = None
            time_offset = next_arrival(1/config['JOB_ARRIVAL_TIME'])
        else: # Prescribed replay
            scheduled_nodes = None
            time_offset = next_arrival(1/config['JOB_ARRIVAL_TIME'])

        trace_quanta = config['TRACE_QUANTA']
            
        if cpu_trace.size > 0 and time_offset >= 0:
            job_info = job_dict(nodes_required = nodes_required, 
                                name = name, 
                                account = account, 
                                cpu_trace = cpu_trace, 
                                gpu_trace = gpu_trace, 
                                ntx_trace = [], 
                                nrx_trace = [],
                                end_state = end_state, 
                                scheduled_nodes = scheduled_nodes,
                                id = job_id, 
                                priority = priority, 
                                submit_time = time_offset, 
                                time_limit = wall_time,
                                trace_quanta=trace_quanta)
            jobs.append(Job(job_info))

    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start, telemetry_end=telemetry_end,
        start_date=telemetry_start_timestamp,
    )


def node_index_to_name(index: int, config: dict):
    """ Converts an index value back to an name string based on system configuration. """
    return f"node{index:04d}"


def cdu_index_to_name(index: int, config: dict):
    return f"cdu{index:02d}"


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    """ Return (row, col) tuple for a cdu index """
    return (0, index) # TODO