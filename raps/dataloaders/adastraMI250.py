"""

    # get the data
    Download `AdastaJobsMI250_15days.parquet` from   https://zenodo.org/records/14007065/files/AdastaJobsMI250_15days.parquet


    # to simulate the dataset
    python main.py -f /path/to/AdastaJobsMI250_15days.parquet --system adastra

    # to replay with different arrival distribution
    python main.py -f /path/to/AdastaJobsMI250_15days.parquet --system adastra --arrival poisson

    # to fast-forward 60 days and replay for 1 day
    python main.py -f /path/to/AdastaJobsMI250_15days.parquet --system adastra -ff 60d -t 1d

    # to analyze dataset
    python -m raps.telemetry -f /path/to/AdastaJobsMI250_15days.parquet --system adastra -v

"""
import uuid
import numpy as np
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
    count_jobs_notOK = 0
    config = kwargs.get('config')
    min_time = kwargs.get('min_time', None)
    arrival = kwargs.get('arrival')
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
        wall_time = jobs_df.loc[jidx, 'run_time']
        if wall_time <= 0:
            print("error wall_time",wall_time)
            continue
        if nodes_required <= 0:
            print("error nodes_required",nodes_required)
            continue
        #wall_time = gpu_trace.size * TRACE_QUANTA # seconds

        if validate:

            node_power = jobs_df.loc[jidx, 'node_power_consumption']
            node_power_array = node_power.tolist()
            node_watts = sum(node_power_array) / (wall_time*nodes_required)
            cpu_trace = node_watts
            gpu_trace = 0.0  # should contain  stddev_node_power when --validate flag is used

        else:
            cpu_power = jobs_df.loc[jidx, 'cpu_power_consumption']
            cpu_power_array = cpu_power.tolist()
            cpu_watts = sum(cpu_power_array) / (wall_time*nodes_required)
            cpu_min_power = config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
            cpu_max_power = config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']


            cpu_util = (cpu_watts/float(config['POWER_CPU_IDLE']) - config['CPUS_PER_NODE']) /  ((float(config['POWER_CPU_MAX']) / float(config['POWER_CPU_IDLE'])) -1.0)    #power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
 #           print("cpu_watts",cpu_watts,"cpu_util",cpu_util)
            cpu_trace = np.maximum(0, cpu_util)

            node_power = (jobs_df.loc[jidx, 'node_power_consumption']).tolist()
            mem_power = (jobs_df.loc[jidx, 'mem_power_consumption']).tolist()
            # Find the minimum length among the three lists
            min_length = min(len(node_power), len(cpu_power), len(mem_power))
            # Slice each list to the minimum length
            node_power = node_power[:min_length]
            cpu_power = cpu_power[:min_length]
            mem_power = mem_power[:min_length]

            gpu_power = (node_power - cpu_power - mem_power
                - ([config['NICS_PER_NODE'] * config['POWER_NIC']]))
            gpu_power_array = gpu_power.tolist()
            gpu_watts = sum(gpu_power_array) / (wall_time*nodes_required)
            gpu_min_power =  config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']
            gpu_max_power =  config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']
            gpu_util = (gpu_watts/float(config['POWER_GPU_IDLE']) - config['GPUS_PER_NODE']) /  ((float(config['POWER_GPU_MAX']) / float(config['POWER_GPU_IDLE'])) -1.0)    #power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
 #           print("gpu_watts",gpu_watts,"gpu_util",gpu_util)
            gpu_trace = np.maximum(0, gpu_util) #gpu_util * GPUS_PER_NODE

        priority = int(jobs_df.loc[jidx, 'priority'])

        end_state = jobs_df.loc[jidx, 'job_state']
        time_start = jobs_df.loc[jidx, 'start_time']
        time_end = jobs_df.loc[jidx, 'end_time']
        diff = time_start - time_zero

        if jid == '*':
            time_offset = max(diff.total_seconds(), 0)
        else:
            # When extracting out a single job, run one iteration past the end of the job
            time_offset = config['UI_UPDATE_FREQ']

        if fastforward:
            time_offset -= fastforward

        if arrival == 'poisson':  # Modify the arrival times of the jobs according to Poisson distribution
            scheduled_nodes = None
            time_offset = next_arrival(1/config['JOB_ARRIVAL_TIME'])
        else:  # Prescribed replay
            scheduled_nodes = (jobs_df.loc[jidx, 'nodes']).tolist()

        if time_offset >= 0 and wall_time > 0:
            job_info = job_dict(nodes_required, name, account, cpu_trace, gpu_trace, [],[],wall_time,
                                end_state, scheduled_nodes, time_offset, job_id, priority)
            jobs.append(job_info)
        else:
            count_jobs_notOK += 1

    print("jobs not added: ", count_jobs_notOK)
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
