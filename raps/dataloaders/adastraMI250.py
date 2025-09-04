"""

    # get the data
    Download `AdastaJobsMI250_15days.parquet` from
    https://zenodo.org/records/14007065/files/AdastaJobsMI250_15days.parquet


    # to simulate the dataset
    python main.py -f /path/to/AdastaJobsMI250_15days.parquet --system adastraMI250

    # to replay with different scheduling policy
    python main.py -f /path/to/AdastaJobsMI250_15days.parquet --system adastraMI250  --policy priority --backfill easy

    # to fast-forward 60 days and replay for 1 day
    python main.py -f /path/to/AdastaJobsMI250_15days.parquet --system adastraMI250 --ff 60d -t 1d

    # to analyze dataset
    python -m raps.telemetry -f /path/to/AdastaJobsMI250_15days.parquet --system adastraMI250 -v

"""
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..job import job_dict, Job
from ..utils import WorkloadData


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
    telemetry_start
    telemetry_end
    """
    count_jobs_notOK = 0
    config = kwargs.get('config')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')

    # Sort jobs dataframe based on values in time_start column, adjust indices after sorting
    jobs_df = jobs_df.sort_values(by='start_time')
    jobs_df = jobs_df.reset_index(drop=True)

    # We only have average power, therefore use the first start time as the start time for the telemetry
    telemetry_start_timestamp = jobs_df['start_time'].min()
    telemetry_end_timestamp = jobs_df['end_time'].max()

    telemetry_start_time = 0
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end_time = int(diff.total_seconds())

    num_jobs = len(jobs_df)
    print("First start time:", telemetry_start_timestamp, "num_jobs", num_jobs)

    jobs = []

    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Jobs"):
        job_id = jobs_df.loc[jidx, 'job_id']
        if not jid == '*':
            if int(jid) == int(job_id):
                print(f'Extracting {job_id} profile')
            else:
                continue
        nodes_required = jobs_df.loc[jidx, 'num_nodes_alloc']
        name = str(uuid.uuid4())[:6]
        account = jobs_df.loc[jidx, 'user_id']

        wall_time = int(jobs_df.loc[jidx, 'run_time'])
        if wall_time <= 0:
            print("error wall_time", wall_time)
            continue
        if nodes_required <= 0:
            print("error nodes_required", nodes_required)
            continue

        if validate:

            node_power = jobs_df.loc[jidx, 'node_power_consumption']
            node_power_array = node_power.tolist()
            node_watts = sum(node_power_array) / (wall_time * nodes_required)
            cpu_trace = node_watts
            gpu_trace = 0.0  # should contain  stddev_node_power when --validate flag is used

        else:
            cpu_power = jobs_df.loc[jidx, 'cpu_power_consumption']
            cpu_power_array = cpu_power.tolist()
            cpu_watts = sum(cpu_power_array) / (wall_time * nodes_required)
            # cpu_min_power = config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']  # Unused
            # cpu_max_power = config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']  # Unused

            cpu_util = (cpu_watts / float(config['POWER_CPU_IDLE']) - config['CPUS_PER_NODE']) \
                / ((float(config['POWER_CPU_MAX']) / float(config['POWER_CPU_IDLE'])) - 1.0)
            # power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)
            # print("cpu_watts",cpu_watts,"cpu_util",cpu_util)

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
            gpu_watts = sum(gpu_power_array) / (wall_time * nodes_required)
            # gpu_min_power = config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']  # Unused
            # gpu_max_power = config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']  # Unused
            gpu_util = (gpu_watts / float(config['POWER_GPU_IDLE']) - config['GPUS_PER_NODE']) \
                / ((float(config['POWER_GPU_MAX']) / float(config['POWER_GPU_IDLE'])) - 1.0)
            # power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            # print("gpu_watts",gpu_watts,"gpu_util",gpu_util)
            gpu_trace = np.maximum(0, gpu_util)

        end_state = jobs_df.loc[jidx, 'job_state']

        priority = int(jobs_df.loc[jidx, 'priority'])

        scheduled_nodes = (jobs_df.loc[jidx, 'nodes']).tolist()

        submit_timestamp = jobs_df.loc[jidx, 'submit_time']
        diff = submit_timestamp - telemetry_start_timestamp
        submit_time = int(diff.total_seconds())

        time_limit = jobs_df.loc[jidx, 'time_limit']  # in seconds

        start_timestamp = jobs_df.loc[jidx, 'start_time']
        diff = start_timestamp - telemetry_start_timestamp
        start_time = int(diff.total_seconds())

        end_timestamp = jobs_df.loc[jidx, 'end_time']
        diff = end_timestamp - telemetry_start_timestamp
        end_time = int(diff.total_seconds())

        if wall_time != end_time - start_time:
            print("wall_time != end_time - start_time")
            print(f"{wall_time} != {end_time - start_time}")
            print(jobs_df[jidx])

        trace_time = wall_time
        trace_start_time = end_time
        trace_end_time = start_time

        if wall_time > 0:
            job_info = job_dict(nodes_required=nodes_required,
                                name=name,
                                account=account,
                                cpu_trace=cpu_trace,
                                gpu_trace=gpu_trace,
                                ntx_trace=[],
                                nrx_trace=[],
                                end_state=end_state,
                                scheduled_nodes=scheduled_nodes,
                                id=job_id,
                                priority=priority,
                                submit_time=submit_time,
                                time_limit=time_limit,
                                start_time=start_time,
                                end_time=end_time,
                                expected_run_time=wall_time,
                                current_run_time=0,
                                trace_time=trace_time,
                                trace_start_time=trace_start_time,
                                trace_end_time=trace_end_time,
                                trace_quanta=None,
                                trace_missing_values=True
                                )
            job = Job(job_info)
            jobs.append(job)
        else:
            count_jobs_notOK += 1

    print("jobs not added: ", count_jobs_notOK)
    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start_time, telemetry_end=telemetry_end_time,
        start_date=telemetry_start_timestamp.tz_localize("UTC"),
    )


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
    node_index = chassis * config['BLADES_PER_CHASSIS'] * \
        config['NODES_PER_BLADE'] + slot * config['NODES_PER_BLADE'] + node
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
