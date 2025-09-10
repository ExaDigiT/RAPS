"""
    Download parquet files from https://zenodo.org/records/11467483

    Note that F-Data doesn't give a list of nodes used, so we set 'scheduled_nodes' to None
    which triggers the scheduler to schedule the nodes itself.

    Also, power in F-Data is only given at node-level. We can use node-level power by
    adding the --validate option.

    The '--arrival poisson' will compute submit times from Poisson distribution, instead of using
    the submit times given in F-Data.

    python main.py --system fugaku -f /path/to/21_04.parquet
    python main.py --system fugaku -f /path/to/21_04.parquet --validate
    python main.py --system fugaku -f /path/to/21_04.parquet --policy priority --backfill easy
"""
import pandas as pd
from tqdm import tqdm
from ..job import job_dict, Job
from ..utils import WorkloadData


def load_data(path, **kwargs):
    """
    Loads data from the given Parquet file path and returns job info.

    Parameters:
    path (str): Path to the Parquet file.

    Returns:
    list: List of job dictionaries.
    """
    # Load the parquet file
    parquet_file = path[0]  # Assuming path is a list containing the path to the parquet file
    df = pd.read_parquet(parquet_file)

    # Process the DataFrame and pass to load_data_from_df
    return load_data_from_df(df, **kwargs)


def load_data_from_df(df, **kwargs):
    """
    Processes DataFrame to extract relevant job information and computes the time offset
    based on the earliest submission time.

    Parameters:
    df (pd.DataFrame): DataFrame containing job information.

    Returns:
    list: List of job dictionaries.
    int: Telemetry Start (in seconds 0)
    int: Telemetry End (in seconds)
    """
    # encrypt_bool = kwargs.get('encrypt')  # Unused
    # arrival = kwargs.get('arrival')  # Unused
    validate = kwargs.get('validate')
    # jid = kwargs.get('jid', '*')  # Unused
    config = kwargs.get('config')

    job_list = []

    # Convert all times to datetime and find the min and max thereof for reference use.
    # Convert 'adt' (submit time) to datetime and find the earliest submission time
    df['adt'] = pd.to_datetime(df['adt'], errors='coerce')
    df['sdt'] = pd.to_datetime(df['sdt'], errors='coerce')
    df['edt'] = pd.to_datetime(df['edt'], errors='coerce')

    # We only have average power therefore we set the earliest telemetry to the earliest start time
    first_start_timestamp = df['sdt'].min()
    last_end_timestamp = df['edt'].max()
    telemetry_start_timestamp = first_start_timestamp
    telemetry_start = 0
    telemetry_end_timestamp = last_end_timestamp
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end = int(diff.total_seconds())

    # Loop through the DataFrame rows to extract job information
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Jobs"):
        nodes_required = row['nnumr'] if 'nnumr' in df.columns else 0
        name = row['jnam'] if 'jnam' in df.columns else 'unknown'
        account = row['usr']

        if validate:
            cpu_trace = row['avgpcon']
            gpu_trace = cpu_trace

        else:
            # cpu_trace = row['perf1'] if 'perf1' in df.columns else 0  # Assuming some performance metric as cpu_trace
            # Total Opts / Total Ops + Idle Ops
            cpu_trace = row['perf1'] / (row['perf1'] + row['perf6']) if 'perf1' in df.columns else 0
            gpu_trace = 0  # Set to 0 as GPU trace is not explicitly provided

        # No network trace

        end_state = row['exit state'] if 'exit state' in df.columns else 'unknown'
        scheduled_nodes = None  # Only nodes_required is in the trace

        job_id = row['jid'] if 'jid' in df.columns else 'unknown'

        priority = row['pri'] if 'pri' in df.columns else 0

        submit_timestamp = pd.to_datetime(row['adt']) if 'adt' in df.columns else - \
            1  # Else job was submitted in the past
        diff = submit_timestamp - telemetry_start_timestamp
        submit_time = int(diff.total_seconds())

        time_limit = int(row['elpl']) if 'elpl' in df.columns else 24 * 60 * 60  # in seconds

        start_timestamp = pd.to_datetime(row['sdt']) if 'sdt' in df.columns else 0
        diff = start_timestamp - telemetry_start_timestamp
        start_time = int(diff.total_seconds())

        end_timestamp = pd.to_datetime(row['edt']) if 'edt' in df.columns else 0
        diff = end_timestamp - telemetry_start_timestamp
        end_time = int(diff.total_seconds())

        wall_time = end_time - start_time
        if end_time < start_time:
            print(f"Job: {i}, skiped end_time < start_time ({end_time} < {start_time})")
            if kwargs.get('debug'):
                print(row)
            continue

        # duration = int(row['duration']) if 'duration' in df.columns else 0
        # if (wall_time != duration):
        #    if abs(wall_time - duration) <= 1:  # offset is often 1
        #        wall_time = min(wall_time,duration)
        #    else:
        #        raise ValueError(f"Duration: {row}")  # Offset can be as large as 15 minutes! Removed.

        # We only have a single average value, set trace times as if we had all.
        trace_time = wall_time
        trace_start_time = start_time
        trace_end_time = end_time
        trace_missing_values = False  # Sane Choice?
        trace_quanta = config['TRACE_QUANTA']

        # Create job dictionary
        job_info = job_dict(
            nodes_required=nodes_required,
            name=name,
            account=account,
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            ntx_trace=[],
            nrx_trace=[],
            trace_quanta=trace_quanta,
            end_state=end_state,
            scheduled_nodes=scheduled_nodes,
            id=job_id,
            priority=priority,
            submit_time=submit_time,
            time_limit=time_limit,
            start_time=start_time,
            end_time=end_time,
            expected_run_time=wall_time,
            trace_time=trace_time,
            trace_start_time=trace_start_time,
            trace_end_time=trace_end_time,
            trace_missing_values=trace_missing_values)
        job = Job(job_info)
        job_list.append(job)

    return WorkloadData(
        jobs=job_list,
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
