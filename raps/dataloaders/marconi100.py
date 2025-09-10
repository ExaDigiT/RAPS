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

    # to replay using differnt schedulers
    python main.py -f /path/to/job_table.parquet --system marconi100 --policy fcfs --backfill easy
    python main.py -f /path/to/job_table.parquet --system marconi100 --policy priority --backfill firstfit

    # to fast-forward 60 days and replay for 1 day
    python main.py -f /path/to/job_table.parquet --system marconi100 --ff 60d -t 1d

    # to analyze dataset
    python -m raps.telemetry -f /path/to/job_table.parquet --system marconi100 -v

"""
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..job import job_dict, Job
from ..utils import power_to_utilization, WorkloadData


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
    # min_time = kwargs.get('min_time', None)  # Unused
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')
    debug = kwargs.get('debug')

    # fastforward = kwargs.get('fastforward')
    # if fastforward:
    #    print(f"fast-forwarding {fastforward} seconds")

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df.sort_values(by='start_time')
    jobs_df = jobs_df.reset_index(drop=True)

    # Dataset has one value from start to finish.
    # Therefore we set telemetry start and end equal to job start and end.
    first_start_timestamp = jobs_df['start_time'].min()
    telemetry_start_timestamp = first_start_timestamp

    last_end_timestamp = jobs_df['end_time'].max()
    telemetry_end_timestamp = last_end_timestamp

    telemetry_start = 0
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end = int(diff.total_seconds())

    num_jobs = len(jobs_df)

    if debug:
        print("num_jobs:", num_jobs)
        print("telemetry_start:", telemetry_start, "simulation_fin", telemetry_end)
        print("telemetry_start_timestamp:", telemetry_start_timestamp,
              "telemetry_end_timestamp", telemetry_end_timestamp)
        print("first_start_timestamp:", first_start_timestamp, "last start timestamp:", jobs_df['time_start'].max())

    jobs = []

    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Jobs"):

        account = jobs_df.loc[jidx, 'user_id']  # or 'user_id' ?
        job_id = jobs_df.loc[jidx, 'job_id']
        # allocation_id =
        nodes_required = jobs_df.loc[jidx, 'num_nodes_alloc']
        end_state = jobs_df.loc[jidx, 'job_state']

        if not jid == '*':
            if int(jid) == int(job_id):
                print(f'Extracting {job_id} profile')
            else:
                continue
        nodes_required = jobs_df.loc[jidx, 'num_nodes_alloc']

        name = str(uuid.uuid4())[:6]  # This generates a random 6 char identifier....

        if validate:
            cpu_power = jobs_df.loc[jidx, 'node_power_consumption'] / jobs_df.loc[jidx, 'num_nodes_alloc']
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
        partition = int(jobs_df.loc[jidx, 'partition'])

        time_limit = jobs_df.loc[jidx, 'time_limit']

        start_timestamp = jobs_df.loc[jidx, 'start_time']
        diff = start_timestamp - telemetry_start_timestamp
        start_time = int(diff.total_seconds())

        end_timestamp = jobs_df.loc[jidx, 'end_time']
        diff = end_timestamp - telemetry_start_timestamp
        end_time = int(diff.total_seconds())

        wall_time = int(jobs_df.loc[jidx, 'run_time'])
        if np.isnan(wall_time):
            wall_time = 0
        if wall_time != (end_time - start_time):
            print("wall_time != (end_time - start_time)")
            print(f"{wall_time} != {(end_time - start_time)}")

        scheduled_nodes = (jobs_df.loc[jidx, 'nodes']).tolist()

        submit_timestamp = jobs_df.loc[jidx, 'submit_time']
        diff = submit_timestamp - telemetry_start_timestamp
        submit_time = int(diff.total_seconds())

        trace_time = gpu_trace.size * config['TRACE_QUANTA']  # seconds
        trace_start_time = 0
        trace_end_time = trace_time
        if wall_time > trace_time:
            missing_trace_time = wall_time - trace_time
            if start_time < 0:
                trace_start_time = missing_trace_time
                trace_end_time = wall_time
            elif end_time > telemetry_end:
                trace_start_time = 0
                trace_end_time = trace_time
            else:
                # Telemetry mission at the end
                trace_start_time = 0
                trace_end_time = trace_time
                trace_missing_values = True

        # What does this do?
        # if jid == '*':
        #    # submit_time = max(submit_time.total_seconds(), 0)
        #    submit_timestamp = jobs_df.loc[jidx, 'submit_time']
        #    diff = submit_timestamp - telemetry_start_timestamp
        #    submit_time = diff.total_seconds()

        # else:
        #    # When extracting out a single job, run one iteration past the end of the job
        #    submit_time = config['UI_UPDATE_FREQ']

        if gpu_trace.size > 0 and (jid == job_id or jid == '*'):  # and time_submit >= 0:

            job_info = job_dict(nodes_required=nodes_required,
                                name=name,
                                account=account,
                                cpu_trace=cpu_trace,
                                gpu_trace=gpu_trace,
                                nrx_trace=[], ntx_trace=[],
                                end_state=end_state,
                                # current_state=current_state,  # PENDING?
                                scheduled_nodes=scheduled_nodes,
                                id=job_id,
                                priority=priority,
                                partition=partition,
                                submit_time=submit_time,
                                time_limit=time_limit,
                                start_time=start_time,
                                end_time=end_time,
                                expected_run_time=wall_time,
                                current_run_time=0,
                                trace_time=trace_time,
                                trace_start_time=trace_start_time,
                                trace_end_time=trace_end_time,
                                trace_quanta=config["TRACE_QUANTA"],
                                trace_missing_values=trace_missing_values)
            job = Job(job_info)
            jobs.append(job)

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
    return (0, index)  # TODO
