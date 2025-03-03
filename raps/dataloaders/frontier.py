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

    telemetry_start
        the first timestep in which the simulation be executed.

    telemetry_end
        the last timestep in which the simulation can be executed.
    ----
    Explanation regarding times:

    The loaded dataframe contains
    a first timestamp with associated data
    and a last timestamp with associated data

    These form the maximum extent of the simuluation time.
    telemetry_start and telemetry_end.

            [                                    ]
            ^                                    ^
            telemetry_start          telemetry_end

    These values form the maximum extent of the simulation.
    Telemetry start == 0! This means that any time before that is negative,
    while anything after this is positive.
    Next is the actual extent of the simulation:

            [                                   ]
                ^                   ^
                simulation_start    simulation_end

    The start of the simulation simulation_start and telemetry_start are only
    the same when fastfoward is 0.
    In general simulation_end and telemetry_end are the same, as this is the
    last time step we can simulate.
    Both simulation_start and _end are set in engine.py

    Additionally, jobs can have started before telemetry_start,
    And can have a recorded ending after simulation_end,
            [                                   ]
    ^                                                ^
    first_start_timestamp           last_end_timestamp

    This means that the time between first_start_timestamp and telemetry_start
    has no associated values in the traces!
    The missing values after simulation_end can be ignored, as the simulatuion
    will have stoped before.

    However, the times before telemetry_start have to be padded to generate
    correct offsets within their data!
    Within the simulation a job's current time is specified as the difference
    between its start_time and the current timestep of the simulation.

    With this each job's
    - submit_time
    - time_limit
    - start_time
    - end_time
    - wall_time (end_time-start_time, actual runtime in seconds)
    - trace_time (lenght of each trace in seconds)
    - trace_start_time (time offset in seconds after which the trace starts)
    - trace_end_time (time offset in seconds after which the trace ends)
    has to be set for use within the simulation

    The values trace_start_time are similar to the telemetry_start and
    telemetry_stop but job specific.

    The returned values are these three:
        - The list of parsed jobs. (as a job_dict)
        - telemetry_start: int (in seconds)
        - telemetry_end: int (in seconds)

    The implementation follows:
    """
    config = kwargs.get('config')
    encrypt_bool = kwargs.get('encrypt')
    arrival = kwargs.get('arrival')
    validate = kwargs.get('validate')
    jid = kwargs.get('jid', '*')
    debug = kwargs.get('debug')

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

    #telemetry_start_timestamp = jobs_df['time_snapshot'].min()  # Earliets time snapshot within the day!
    telemetry_start_timestamp = jobprofile_df['timestamp'].min()  # Earliets time snapshot within the day!
    #telemetry_end_timestamp = jobs_df['time_snapshot'].max()  # This time has nothing to do with the jobs!
    telemetry_end_timestamp = jobprofile_df['timestamp'].max()  # Earliets time snapshot within the day!

    # Time that can be simulated # Take earliest time as baseline reference
    telemetry_start = 0  # second 0 of the simulation
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end = int(diff.total_seconds())

    first_start_timestamp = jobs_df['time_start'].min()
    diff = first_start_timestamp - telemetry_start_timestamp
    first_start = int(diff.total_seconds())  # negative seconds or 0

    num_jobs = len(jobs_df)
    if debug:
        print("num_jobs:", num_jobs)
        print("telemetry_start:", telemetry_start, "simulation_fin", telemetry_end)
        print("telemetry_start_timestamp:", telemetry_start_timestamp, "telemetry_end_timestamp", telemetry_end_timestamp)
        print("first_start_timestamp:",first_start_timestamp, "last start timestamp:", jobs_df['time_start'].max())

    jobs = []
    # Map dataframe to job state. Add results to jobs list
    for jidx in tqdm(range(num_jobs - 1), total=num_jobs, desc="Processing Jobs"):

        # user = jobs_df.loc[jidx, 'user']
        account = jobs_df.loc[jidx, 'account']
        job_id = jobs_df.loc[jidx, 'job_id']
        allocation_id = jobs_df.loc[jidx, 'allocation_id']
        nodes_required = jobs_df.loc[jidx, 'node_count']
        end_state = jobs_df.loc[jidx, 'state_current']
        name = jobs_df.loc[jidx, 'name']
        if encrypt_bool:
            name = encrypt(name)

        if validate:
            cpu_power = jobprofile_df[jobprofile_df['allocation_id'] \
                                      == allocation_id]['mean_node_power']
            cpu_trace = cpu_power.values
            gpu_trace = cpu_trace

        else:
            cpu_power = jobprofile_df[jobprofile_df['allocation_id'] \
                                      == allocation_id]['sum_cpu0_power']
            cpu_power_array = cpu_power.values
            cpu_min_power = nodes_required * config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
            cpu_max_power = nodes_required * config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']
            cpu_util = power_to_utilization(cpu_power_array, cpu_min_power, cpu_max_power)  # Will be negative! as cpu_power_array[i] can be smaller than cpu_min_power
            cpu_trace = cpu_util * config['CPUS_PER_NODE']

            gpu_power = jobprofile_df[jobprofile_df['allocation_id'] \
                                      == allocation_id]['sum_gpu_power']
            gpu_power_array = gpu_power.values

            gpu_min_power = nodes_required * config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']
            gpu_max_power = nodes_required * config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']
            gpu_util = power_to_utilization(gpu_power_array, gpu_min_power, gpu_max_power)
            gpu_trace = gpu_util * config['GPUS_PER_NODE']

        # Set any NaN values in cpu_trace and/or gpu_trace to zero
        cpu_trace[np.isnan(cpu_trace)] = 0
        gpu_trace[np.isnan(gpu_trace)] = 0

        # Times:
        submit_timestamp = jobs_df.loc[jidx, 'time_submission']
        diff = submit_timestamp - telemetry_start_timestamp
        submit_time = diff.total_seconds()

        time_limit = jobs_df.loc[jidx, 'time_limit']  # timelimit in seconds

        start_timestamp = jobs_df.loc[jidx, 'time_start']
        diff = start_timestamp - telemetry_start_timestamp
        start_time = diff.total_seconds()

        end_time_timestamp = jobs_df.loc[jidx, 'time_end']
        diff = end_time_timestamp - telemetry_start_timestamp
        end_time = diff.total_seconds()

        wall_time = end_time - start_time
        if np.isnan(wall_time):
            wall_time = 0

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
                print(f"Job: {job_id} {start_time} - {end_time}!")
                raise ValueError("Missing values not at start nor end.")

        xnames = jobs_df.loc[jidx, 'xnames']
        # Don't replay any job with an empty set of xnames
        if '' in xnames:
            continue

        if arrival == 'poisson':  # Modify the arrival times of the jobs according to Poisson distribution
            scheduled_nodes = None
            submit_time = next_arrival(1 / config['JOB_ARRIVAL_TIME'])
            start_time = None  # ?
            end_time = None  # ?
            priority = aging_boost(nodes_required)

        else:  # Prescribed replay
            scheduled_nodes = []
            # priority = 0  # not used for replay
            priority = aging_boost(nodes_required)
            for xname in xnames:
                indices = xname_to_index(xname, config)
                scheduled_nodes.append(indices)

        # Throw out jobs that are not valid!
        if gpu_trace.size == 0:
            print("ignoring job b/c zero trace:", jidx, submit_time, start_time, nodes_required)
            continue  # SKIP!
        if end_time < telemetry_start:
            # raise ValueError("Job ends before frist recorded telemetry entry:",job_id, "start:", start_time,"end:",end_time, " Telemetry: ", len(gpu_trace), "entries.")
            print("Job ends before frist recorded telemetry entry:",job_id, "start:", start_time,"end:",end_time, " Telemetry: ", len(gpu_trace), "entries.")
            continue  # SKIP!
        if start_time > telemetry_end:
            # raise ValueError("Job starts after last recorded telemetry entry:",job_id, "start:", start_time,"end:",end_time, " Telemetry: ", len(gpu_trace), "entries.")
            print("Job starts after last recorded telemetry entry:",job_id, "start:", start_time,"end:",end_time, " Telemetry: ", len(gpu_trace), "entries.")
            continue  # SKIP!

        if gpu_trace.size > 0 and (jid == job_id or jid == '*'):  # and time_submit >= 0:
            job_info = job_dict(nodes_required, name, account, cpu_trace, gpu_trace, [], [],
                                end_state, scheduled_nodes,
                                job_id, priority,  # partition missing
                                submit_time=submit_time, time_limit=time_limit,
                                start_time=start_time, end_time=end_time,
                                wall_time=wall_time, trace_time=trace_time,
                                trace_start_time=trace_start_time, trace_end_time=trace_end_time)
            jobs.append(job_info)

    return jobs, telemetry_start, telemetry_end


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
