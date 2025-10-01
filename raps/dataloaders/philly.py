import os
import glob
import json
import csv
import pandas as pd
import warnings

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from raps.job import job_dict, Job
from raps.utils import WorkloadData

DATE_FORMAT_STR = "%Y-%m-%d %H:%M:%S"
DEFAULT_START = "2017-10-03T00:00"
DEFAULT_END = "2017-10-04T00:00"

def to_epoch(ts_str):
    if ts_str is None:
        return None
    if isinstance(ts_str, (int, float)):
        return int(ts_str)
    if "T" in ts_str:
        dt = datetime.fromisoformat(ts_str)
    else:
        dt = datetime.strptime(ts_str, DATE_FORMAT_STR)
    return int(dt.timestamp())

def parse_timestamp(val):
    """
    Convert Philly job log timestamps to datetime.
    Handles integers (epoch) and strings with PST/PDT.
    Returns datetime or None.
    """
    if val is None or val == "None":
        return None
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(int(val), tz=timezone.utc).replace(tzinfo=None)
    if isinstance(val, str):
        val = val.replace(" PST", "").replace(" PDT", "")
        try:
            return datetime.strptime(val, DATE_FORMAT_STR).replace(tzinfo=None)
        except ValueError:
            return None
    return None

def load_traces_by_day(trace_dir, start_dt, end_dt, colname):
    """Load CPU or GPU traces between start_dt and end_dt."""
    frames = []
    current = start_dt.date()

    while current <= end_dt.date():
        daily_file = os.path.join(trace_dir, f"{current}.csv")
        if os.path.exists(daily_file):
            df = pd.read_csv(
                daily_file,
                names=["time", "machineId", colname],  # no header in daily CSVs
                dtype={"machineId": str, colname: str}, # avoid DtypeWarning
            )

            # Normalize time column (strip PST/PDT, parse datetime)
            df["time"] = df["time"].str.replace(" PST", "").str.replace(" PDT", "")
            df["time"] = pd.to_datetime(df["time"], errors="coerce", format=DATE_FORMAT_STR)

            # Convert util column to numeric (NA/invalid → NaN)
            df[colname] = pd.to_numeric(df[colname], errors="coerce")

            frames.append(df)
        else:
            print(f"⚠ No trace file for {current}")
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["time", "machineId", colname])

    return pd.concat(frames, ignore_index=True)

def parse_date(s):
    if not s or s == "None":
        return None
    # strip possible timezone labels like "PST"/"PDT"
    s = s.replace(" PST", "").replace(" PDT", "")
    return datetime.strptime(s, DATE_FORMAT_STR)

def load_data(files, **kwargs):
    """
    Load Philly trace into ExaDigiT Job objects.

    Args:
        files (list[str]): A list with one directory path (e.g., ['/opt/data/philly/trace-data']).

    Returns:
        list[Job]
    """
    # extract --start from kwargs
    start_ts = to_epoch(kwargs.get("start", DEFAULT_START))
    end_ts = to_epoch(kwargs.get("end", DEFAULT_END))

    assert len(files) == 1, "Expecting a single directory path"
    trace_dir = files[0]
    gpu_trace_dir = os.path.join(files[0], "dist", "gpu_by_day")
    config = kwargs.get('config')
    gpus_per_node = config.get("GPUS_PER_NODE")
    if gpus_per_node is None:
        raise ValueError("Must pass gpus_per_node (2 or 8)")

    # --- 1. Machine list ---
    machine_file = os.path.join(trace_dir, "cluster_machine_list")
    machines = {}
    with open(machine_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["machineId"]
            machines[mid] = {
                "num_gpus": int(row[" number of GPUs"]),
                "gpu_mem": row[" single GPU mem"].strip()
            }

    partition_machines = {
        mid: info for mid, info in machines.items()
        if info["num_gpus"] == gpus_per_node
    }

    # Build node → index mapping for this partition
    node_mapping = {mid: idx for idx, mid in enumerate(sorted(partition_machines.keys()))}
    max_nodes = len(node_mapping)

    # Assign partition ID (e.g. 0 for 2-GPU, 1 for 8-GPU)
    partition_id = 0 if gpus_per_node == 2 else 1

    # --- 2. CPU util ---
    cpu_file = os.path.join(trace_dir, "cluster_cpu_util")
    cpu_util = pd.read_csv(cpu_file)
    cpu_util["time"] = cpu_util["time"].str.replace(" PST","").str.replace(" PDT","")
    cpu_util["time"] = cpu_util["time"].apply(parse_date)

    # --- 3. GPU util ---
    #PDT = timezone(timedelta(hours=-7))
    #start_dt = datetime.fromtimestamp(start_ts, tz=PDT)
    #end_dt   = datetime.fromtimestamp(end_ts, tz=PDT)
    start_dt = datetime.fromtimestamp(start_ts)  # naive datetime
    end_dt   = datetime.fromtimestamp(end_ts)

    cpu_trace_dir = os.path.join(trace_dir, "dist", "cpu_by_day")
    gpu_trace_dir = os.path.join(trace_dir, "dist", "gpu_by_day")

    # --- 4. Job log ---
    job_file = os.path.join(trace_dir, "cluster_job_log")
    with open(job_file) as f:
        job_log = json.load(f)

    # Filter job_log to only jobs matching the partition's gpus_per_node
    if gpus_per_node is not None:
        filtered_log = []
        for raw in job_log:
            attempts = raw.get("attempts", [])
            if attempts and "detail" in attempts[0]:
                # Count GPUs from the first detail
                gpus = sum(len(detail.get("gpus", [])) for detail in attempts[0]["detail"])
                if gpus > 0 and (gpus % gpus_per_node == 0):
                    filtered_log.append(raw)
        job_log = filtered_log

    # --- First pass: find earliest submit time ---
    start_ts = None
    for raw in job_log:
        submitted = raw.get("submitted_time")
        if submitted is None or submitted == "None":
            continue

        # Philly uses either string dates or epoch ints
        if isinstance(submitted, (int, float)):
            t = int(submitted)
        else:
            t = parse_date(submitted).timestamp()

        if start_ts is None or t < start_ts:
            start_ts = t

    if start_ts is None:
        raise ValueError("No valid submitted_time found in Philly traces")

    # --- Second pass: build jobs ---
    jobs_list = []
    for raw in tqdm(job_log, desc="Building Job objects"):
        jobid = raw.get("jobid")
        user = raw.get("user")
        status = raw.get("status")

        # Submitted time
        submitted = raw.get("submitted_time")
        if isinstance(submitted, (int, float)):
            submitted = datetime.fromtimestamp(int(submitted))
        else:
            submitted = parse_date(submitted)

        attempts = raw.get("attempts", [])
        start, end = None, None
        if attempts:
            st = attempts[0].get("start_time")
            et = attempts[-1].get("end_time")

            if isinstance(st, (int, float)):
                start = datetime.fromtimestamp(int(st))
            elif st:
                start = parse_date(st)

            if isinstance(et, (int, float)):
                end = datetime.fromtimestamp(int(et))
            elif et:
                end = parse_date(et)

        wall_time = None
        if start and end:
            wall_time = (end - start).total_seconds()

        # Which machines did this job run on?
        machine_ids, gpus = [], 0
        if attempts and "detail" in attempts[0]:
            for detail in attempts[0]["detail"]:
                mid = detail["ip"]
                machine_ids.append(mid)
                gpus += len(detail.get("gpus", []))
        
        num_nodes = len(machine_ids)
        gpus_per_node = gpus // num_nodes

        # --- absolute datetimes (used for filtering traces) ---
        submitted_dt = parse_timestamp(raw.get("submitted_time"))

        # Clamp to global CLI window - this should be fixed later to include the actual
        # trace start and end times (trace_start_time? and trace_end_time?) 
        #job_start = max(start, start_dt) if start else start_dt
        #job_end   = min(end, end_dt) if end else end_dt
        job_start = start
        job_end = end

        # CPU utilization traces
        cpu_trace = load_traces_by_day(cpu_trace_dir, job_start, job_end, "cpu_util")

        mask = (
            (cpu_trace["machineId"].isin(machine_ids)) &
            (cpu_trace["time"] >= start) &
            (cpu_trace["time"] <= end)
        )
        job_cpu = cpu_trace.loc[mask].copy()

        # Aggregate across machines if >1 machine
        if len(machine_ids) > 1:
            job_cpu = job_cpu.groupby("time")["cpu_util"].mean().reset_index()

        # Convert from percentage to fraction
        job_cpu_trace = (job_cpu["cpu_util"].to_numpy() * 0.01).tolist()

        # Extract GPU utilization traces
        gpu_trace = load_traces_by_day(gpu_trace_dir, job_start, job_end, "gpu_util")

        mask = (
            (gpu_trace["machineId"].isin(machine_ids)) &
            (gpu_trace["time"] >= start) &
            (gpu_trace["time"] <= end)
        )
        # Convert traces from percent to fraction of gpus_per_node, e.g., 8 gpus at 100% is 8, at 50% is 4, etc.
        job_gpu = gpu_trace.loc[mask].copy()

        # Aggregate across machines if >1 machine
        if len(machine_ids) > 1:
            job_gpu = job_gpu.groupby("time")["gpu_util"].mean().reset_index()

        job_gpu_trace = (job_gpu["gpu_util"].to_numpy() * 0.01 * gpus_per_node).tolist()


        if machine_ids:
            # Shift times relative to start_ts
            submit_time = submitted.timestamp() - start_ts if submitted else None
            start_time = start.timestamp() - start_ts if start else None
            end_time = end.timestamp() - start_ts if end else None

            if not submit_time or not start_time or not end_time:
                warnings.warn(
                    f"skipped {jobid} b/c missing submit_time, start_time, or end_time",
                    UserWarning
                )
       
            scheduled_nodes = [node_mapping[mid] for mid in machine_ids if mid in node_mapping]

            if submit_time and start_time and end_time: 

                job = job_dict(
                    id=jobid,
                    name=f"philly-{jobid}",
                    account=user if user else "unknown",

                    nodes_required=len(machine_ids),
                    partition=partition_id,
                    priority=0,

                    cpu_cores_required=1,
                    gpu_units_required=gpus_per_node,

                    end_state=status,
                    scheduled_nodes=scheduled_nodes,

                    cpu_trace=job_cpu_trace,
                    gpu_trace=job_gpu_trace,
                    ntx_trace=None,
                    nrx_trace=None,

                    submit_time=submit_time,
                    start_time=start_time,
                    end_time=end_time,
                    time_limit=0,
                    expected_run_time=wall_time if wall_time else 0,
                    current_run_time=0,
                    trace_time=None,
                    trace_start_time=None,
                    trace_end_time=None,
                    trace_quanta=None,
                    trace_missing_values=False,
                    downscale=1
                )
                jobs_list.append(Job(job))

            print(job)

#        if len(jobs_list) >= 5: 
#            break

        # Find max end timestamp across jobs, relative to first job
        end_ts = max(j.end_time for j in jobs_list if j.end_time is not None)

    # Absolute end_ts
    end_ts = start_ts + end_ts

    return WorkloadData(
        jobs=jobs_list,
        telemetry_start=0, telemetry_end=int(end_ts - start_ts),
        start_date=datetime.fromtimestamp(start_ts, timezone.utc),
    )
