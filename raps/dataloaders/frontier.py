"""
    Note: Frontier telemetry data is not publicly available.

    # To simulate
    DATEDIR="date=2024-01-18"
    DPATH=/path/to/data
    python main.py -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR

    # To analyze the data
    python -m raps.telemetry -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR
"""
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..job import job_dict, Job
from ..utils import power_to_utilization, encrypt, WorkloadData


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
    if kwargs.get("live") is True:
        return load_live_data()
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
    - start_time  # Maybe Null
    - end_time  # Maybe Null
    - expected_run_time (end_time - start_time)  # Maybe Null
    - current_run_time (How long did the job run already, when loading)  # Maybe zero
    - trace_time (lenght of each trace in seconds)  # Maybe Null
    - trace_start_time (time offset in seconds after which the trace starts)  # Maybe Null
    - trace_end_time (time offset in seconds after which the trace ends)  # Maybe Null
    - trace_quanta (job's associated trace quanta, to correctly replay with different trace quanta) # Maybe Null
    has to be set for use within the simulation

    The values trace_start_time are similar to the telemetry_start and
    telemetry_stop but may different due to missing data, for each job.

    The returned values are these three:
        - The list of parsed jobs. (as a Job object)
        - telemetry_start: int (in seconds)
        - telemetry_end: int (in seconds)

    The implementation follows:
    """
    config = kwargs.get('config')
    encrypt_bool = kwargs.get('encrypt')
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

    # telemetry_start_timestamp = jobs_df['time_snapshot'].min()  # Earliets time snapshot within the day!
    telemetry_start_timestamp = jobprofile_df['timestamp'].min()  # Earliets time snapshot within the day!
    # telemetry_end_timestamp = jobs_df['time_snapshot'].max()  # This time has nothing to do with the jobs!
    telemetry_end_timestamp = jobprofile_df['timestamp'].max()  # Earliets time snapshot within the day!

    # Time that can be simulated # Take earliest time as baseline reference
    telemetry_start = 0  # second 0 of the simulation
    diff = telemetry_end_timestamp - telemetry_start_timestamp
    telemetry_end = int(diff.total_seconds())

    first_start_timestamp = jobs_df['time_start'].min()
    diff = first_start_timestamp - telemetry_start_timestamp
    # first_start = int(diff.total_seconds())  # negative seconds or 0  # Unused

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
            # Will be negative! as cpu_power_array[i] can be smaller than cpu_min_power
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
        if not start_time <= end_time or np.isnan(end_time):
            continue  # Start_time is not smaller than end_time or is not valid
            # Skip entry.

        expected_run_time = end_time - start_time
        current_run_time = 0  # Check if we the job may  may be runninghave wall time of the jobs

        trace_quanta = config['TRACE_QUANTA']
        trace_time = gpu_trace.size * trace_quanta  # seconds

        trace_start_time = 0
        trace_end_time = trace_time
        if expected_run_time > trace_time:
            missing_trace_time = int(expected_run_time - trace_time)
            trace_missing_values = True
            if start_time < 0:
                trace_start_time = missing_trace_time
                trace_end_time = expected_run_time
            elif end_time > telemetry_end:
                trace_start_time = 0
                trace_end_time = trace_time
            else:
                print(f"Job: {job_id} {end_state} {start_time} - {end_time}, "
                      f"Trace: {trace_start_time} - {trace_end_time}, "
                      f"Missing: {missing_trace_time}!")
        else:
            trace_missing_values = False

        xnames = jobs_df.loc[jidx, 'xnames']
        # Don't replay any job with an empty set of xnames
        if '' in xnames:
            continue

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
            # raise ValueError("Job ends before frist recorded telemetry entry:",
            #                  job_id, "start:", start_time,"end:",end_time,
            #                  " Telemetry: ", len(gpu_trace), "entries.")
            print("Job ends before frist recorded telemetry entry:", job_id, "start:",
                  start_time, "end:", end_time, " Telemetry: ", len(gpu_trace), "entries.")
            continue  # SKIP!
        if start_time > telemetry_end:
            # raise ValueError("Job starts after last recorded telemetry entry:",
            #                  job_id, "start:", start_time,"end:",end_time,
            #                  " Telemetry: ", len(gpu_trace), "entries.")
            print("Job starts after last recorded telemetry entry:", job_id, "start:",
                  start_time, "end:", end_time, " Telemetry: ", len(gpu_trace), "entries.")
            continue  # SKIP!

        if gpu_trace.size > 0 and (jid == job_id or jid == '*'):  # and time_submit >= 0:
            job_info = job_dict(
                nodes_required=nodes_required,
                name=name,
                account=account,
                cpu_trace=cpu_trace,
                gpu_trace=gpu_trace,
                nrx_trace=None,
                ntx_trace=None,
                end_state=end_state,
                scheduled_nodes=scheduled_nodes,
                id=job_id,
                priority=priority,  # partition missing
                submit_time=submit_time,
                time_limit=time_limit,
                start_time=start_time,
                end_time=end_time,
                expected_run_time=expected_run_time,
                current_run_time=current_run_time,
                trace_time=trace_time,
                trace_start_time=trace_start_time, trace_end_time=trace_end_time,
                trace_quanta=trace_quanta, trace_missing_values=trace_missing_values)

            job = Job(job_info)
            jobs.append(job)
    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start,
        telemetry_end=telemetry_end,
        start_date=telemetry_start_timestamp,
    )


def load_live_data(**kwargs):
    """ Load Slurm Live data using pyslurm """
    jobs = list()
    telemetry_start = int(time.time())  # This is now! get unix time
    telemetry_start = 1755721300
    if hasattr(kwargs, 'time'):
        time_to_sim = kwargs.get('time')  # Should be specified .
        assert isinstance(time_to_sim, int)
    else:
        time_to_sim = 14 * 24 * 60 * 60   # or we simulate 2 weeks.
    telemetry_end = telemetry_start + time_to_sim

    total_partitions = 0
    partition_dict = dict()

    import pyslurm  # noqa
    # Local Tests
    # filename = "something/something/pyslurm.dump"
    # with open(filename, 'r') as f:
    #   s = f.read()
    #   data = ast.literal_eval(s)
    #
    data = pyslurm.job().get()

    for jidx, jdata in data.items():
        if jdata['job_state'] == "COMPLETED" \
                or jdata['job_state'] == "CANCELLED":
            continue
        if jdata['job_state'] == "TIMEOUT" \
                or jdata['job_state'] == "FAILED":
            if jdata['requeue'] is False:
                continue

        # if jidx == XXX:
        #    print(jdata)
        #    exit()
        # Picking the useful ones from the 110 features: Leaving the rest for potential changes
        account = jdata['account']
        # 'accrue_time': String  = 'Unknown',
        # 'admin_comment': String,
        # 'alloc_node': String = 'login08',
        # 'alloc_sid':  int
        # 'array_job_id': None,
        # 'array_task_id': None,
        # 'array_task_str': None,
        # 'het_job_id': None,
        # 'het_job_id_set': None,
        # 'het_job_offset': None,
        # 'array_max_tasks': None,
        # 'assoc_id': int,
        # 'batch_flag': int,
        # 'batch_features': None,
        # 'batch_host': None,
        # 'billable_tres': float,
        # 'bitflags': int,
        # 'boards_per_node': int,
        # 'burst_buffer': None,
        # 'burst_buffer_state': None,
        # 'command': String,
        # 'comment': None,
        # 'contiguous': bool,
        # 'core_spec': int,
        # 'cores_per_socket': int,
        # 'cpus_per_task': int,
        # 'cpus_per_tres': None,
        # 'cpu_freq_gov': int,
        # 'cpu_freq_max': int,
        # 'cpu_freq_min': int,
        # 'dependency': None,
        # 'derived_ec': String,
        # 'eligible_time': int,
        # 'end_time': int,
        # 'exc_nodes': [],
        # 'exit_code': String,
        # 'features': [],
        # 'group_id': int,
        job_id = jdata['job_id']
        current_state = jdata['job_state']
        end_state = None
        # 'last_sched_eval': String # e.g. '2013-02-31T14:29:09',
        # 'licenses': {},
        # 'max_cpus': int,
        # 'max_nodes': int,
        # 'mem_per_tres': None,
        name = jdata['name']
        # 'network': None,
        # 'nodes': None,
        # 'nice': 0,
        # 'ntasks_per_core': int,
        # 'ntasks_per_core_str': String
        # 'ntasks_per_node': int,
        # 'ntasks_per_socket': int,
        # 'ntasks_per_socket_str': String,
        # 'ntasks_per_board': 0,
        # 'num_cpus': int,
        nodes_required: int = jdata['num_nodes']
        # 'num_tasks': 49152,
        # 'partition': String,  # e.g.'batch',
        if jdata['partition'] in partition_dict:
            pass
        else:
            partition_dict[jdata['partition']] = total_partitions
            total_partitions += 1
        partition = partition_dict[jdata['partition']]
        # 'mem_per_cpu': bool,
        # 'min_memory_cpu': None,
        # 'mem_per_node': bool,
        # 'min_memory_node': int,
        # 'pn_min_memory': int,
        # 'pn_min_cpus': int,
        # 'pn_min_tmp_disk': int,
        priority = jdata['priority']
        # 'profile': int,
        # 'qos': String  # e.g. 'normal',
        # 'reboot': int,
        scheduled_nodes_str_list = jdata['req_nodes']  # Explicitly requested nodes  # Missmatch between slurm and raps
        scheduled_nodes = []
        for n in scheduled_nodes_str_list:
            scheduled_nodes = int(n[8:])
        # Do we need to reintroduce a list of explicitly required nodes? This is currently handled by setting the
        # scheduled_nodes before the scheduler modifies this list
        # 'req_switch': int,
        # 'requeue': bool,
        # 'resize_time': int,
        # 'restart_cnt': int,
        # 'resv_name': None,
        # 'run_time': int,  # ??
        # 'run_time_str': String,
        # 'sched_nodes': None,
        # 'selinux_context': None,
        # 'shared': String,
        # 'sockets_per_board': int,
        # 'sockets_per_node': int,
        if current_state == "RUNNING":
            start_time = jdata['start_time']
            end_time = None
            current_run_time = jdata['run_time']
        else:
            start_time = None
            end_time = None
            current_run_time = jdata['run_time']  # ??
            if jdata['job_state'] == "TIMEOUT":
                if jdata['requeue'] is False:
                    current_run_time = 0  # ??
            elif jdata['job_state'] == "COMPLETING":
                if jdata['requeue'] is False:
                    current_run_time = 0  # ??
            else:
                assert current_run_time == 0, "Check if any other value occurs and should be handled! " \
                                              f"current_run_time:{current_run_time}" \
                                              f"\njdata:\n{jdata}"
        expected_run_time = None
        # 'state_reason': String  # e.g. 'JobHeldUser',
        # 'std_err': String,
        # 'std_in': String,
        # 'std_out': String,
        submit_time = jdata['submit_time']  # int,  Unix Time!
        # 'suspend_time': int,
        # 'system_comment': None,
        # 'time_limit': e.g. 570,  # in minutes!
        time_limit = jdata['time_limit'] * 60  # needed in seconds
        # 'time_limit_str': '0-09:30:00',
        # 'time_min': int,
        # 'threads_per_core': int,
        # 'tres_alloc_str': None,
        # 'tres_bind': None,
        # 'tres_freq': None,
        # 'tres_per_job': None,
        # 'tres_per_node': None,
        # 'tres_per_socket': None,
        # 'tres_per_task': None,
        # 'tres_req_str': String,
        account = jdata['user_id']  # int for slurm, may be String in raps and conversion works. ...
        # 'wait4switch': int,
        # 'wckey': None,
        # 'work_dir': String
        # 'cpus_allocated': dict,
        # 'cpus_alloc_layout': dict
        cpu_trace = None  # To be determined by a model!
        gpu_trace = None
        trace_time = None
        trace_start_time = None
        trace_end_time = None
        trace_quanta = None
        trace_missing_values = None
        job_info = job_dict(
            nodes_required=nodes_required,
            name=name,
            account=account,
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            nrx_trace=None,
            ntx_trace=None,
            current_state=current_state,
            end_state=end_state,
            scheduled_nodes=scheduled_nodes,
            id=job_id,
            priority=priority,  # partition missing
            partition=partition,
            submit_time=submit_time, time_limit=time_limit,
            start_time=start_time, end_time=end_time,
            expected_run_time=expected_run_time,
            current_run_time=current_run_time,
            trace_time=trace_time,
            trace_start_time=trace_start_time, trace_end_time=trace_end_time,
            trace_quanta=trace_quanta, trace_missing_values=trace_missing_values)
        job = Job(job_info)
        jobs.append(job)

    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start,
        telemetry_end=telemetry_end,
        start_date=datetime.fromtimestamp(telemetry_start, timezone.utc),
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
