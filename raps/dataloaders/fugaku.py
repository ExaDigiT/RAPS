"""
    Download parquet files from https://zenodo.org/records/11467483

    Note that F-Data doesn't give a list of nodes used, so we set 'scheduled_nodes' to None
    which triggers the scheduler to schedule the nodes itself.

    Also, power in F-Data is only given at node-level. We can use node-level power by
    adding the --validate option.

    The '--reschedule poisson' will compute submit times from Poisson distribution, instead of using
    the submit times given in F-Data.

    python main.py --system fugaku -f /path/to/21_04.parquet --reschedule poisson --validate

"""
import pandas as pd
from tqdm import tqdm
from ..job import job_dict
from ..utils import next_arrival


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
    """
    encrypt_bool = kwargs.get('encrypt')
    fastforward = kwargs.get('fastforward')
    reschedule = kwargs.get('reschedule')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')
    config = kwargs.get('config')
    min_time = kwargs.get('min_time', None)

    if fastforward: print(f"fast-forwarding {fastforward} seconds")

    job_list = []

    # Convert 'adt' (submit time) to datetime and find the earliest submission time
    df['adt'] = pd.to_datetime(df['adt'], errors='coerce')
    if not min_time:
        min_time = df['adt'].min()

    # Loop through the DataFrame rows to extract job information
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Jobs"):
        nodes_required = row['nnumr'] if 'nnumr' in df.columns else 0
        account = row['usr']
        name = row['jnam'] if 'jnam' in df.columns else 'unknown'

        if validate:
            cpu_trace = row['avgpcon']
            gpu_trace = cpu_trace

        else:
            cpu_trace = row['perf1'] if 'perf1' in df.columns else 0  # Assuming some performance metric as cpu_trace
            gpu_trace = 0  # Set to 0 as GPU trace is not explicitly provided

        wall_time = row['duration'] if 'duration' in df.columns else 0
        end_state = row['exit state'] if 'exit state' in df.columns else 'unknown'
        #scheduled_nodes = row['nnuma'] if 'nnuma' in df.columns else 0
        scheduled_nodes = None
        submit_time = row['adt'] if 'adt' in df.columns else min_time
        if reschedule == 'poisson':  # Let the scheduler reschedule the jobs
            time_offset = next_arrival(1/config['JOB_ARRIVAL_TIME'])
        else:
            time_offset = (submit_time - min_time).total_seconds()  # Compute time offset in seconds

        job_id = row['jid'] if 'jid' in df.columns else 'unknown'
        priority = row['pri'] if 'pri' in df.columns else 0

        # Create job dictionary
        job_info = job_dict(
            nodes_required=nodes_required,
            name=name,
            account=account,
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            ntx_trace=[],
            nrx_trace=[],
            wall_time=wall_time,
            end_state=end_state,
            scheduled_nodes=scheduled_nodes,
            time_offset=time_offset,
            job_id=job_id,
            priority=priority
        )

        job_list.append(job_info)

    return job_list


def node_index_to_name(index: int, config: dict):
    """ Converts an index value back to an name string based on system configuration. """
    return f"node{index:04d}"


def cdu_index_to_name(index: int, config: dict):
    return f"cdu{index:02d}"


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    """ Return (row, col) tuple for a cdu index """
    return (0, index) # TODO
