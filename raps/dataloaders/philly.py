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

def load_gpu_traces_by_dayXX(gpu_trace_dir, machine_ids, job_start_dt, job_end_dt):
    """
    Load GPU utilization for specific machines and time range,
    using preprocessed per-day CSVs (gpu_by_day/).
    """
    dfs = []
    current = job_start_dt.date()
    while current <= job_end_dt.date():
        day_file = os.path.join(gpu_trace_dir, f"{current}.csv")
        if os.path.exists(day_file):
            df = pd.read_csv(
                day_file,
                names=["time", "machineId", "gpu_util"],
                parse_dates=["time"],
                on_bad_lines="skip"
            )
            df = df[df["machineId"].isin(machine_ids)]
            df = df[(df["time"] >= job_start_dt) & (df["time"] <= job_end_dt)]
            if not df.empty:
                dfs.append(df)
        current += timedelta(days=1)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["time", "machineId", "gpu_util"])


def load_gpu_traces_by_day(trace_dir, start_dt, end_dt):
    """Load GPU traces only for the days between start_dt and end_dt."""
    gpu_dir = os.path.join(trace_dir, "dist/gpu_by_day")
    frames = []

    current = start_dt.date()
    while current <= end_dt.date():
        daily_file = os.path.join(gpu_dir, f"{current}.csv")
        if os.path.exists(daily_file):
            df = pd.read_csv(
                daily_file,
                names=["time", "machineId", "gpu_util"],  # no header in daily CSVs
                parse_dates=["time"]
            )
            frames.append(df)
        else:
            print(f"⚠ No trace file for {current}")
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["time", "machineId", "gpu_util"])

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
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt   = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    gpu_trace_dir = os.path.join(trace_dir, "dist", "gpu_by_day")

    # For each job:
    gpu_trace = load_gpu_traces_by_day(gpu_trace_dir, start_dt, end_dt)
    job_gpu = load_gpu_traces_by_day(gpu_trace_dir, start_dt, end_dt)

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
    for raw in tqdm(job_log[:1000], desc="Building Job objects"):
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

        # CPU utilization traces
        if machine_ids and start and end:
            mask = (
                cpu_util["machine_id"].isin(machine_ids) &
                (cpu_util["time"] >= start) &
                (cpu_util["time"] <= end)
            )
            job_cpu = cpu_util.loc[mask].copy()

            # Aggregate across machines if >1 machine
            if len(machine_ids) > 1:
                job_cpu = job_cpu.groupby("time")["cpu_util"].mean().reset_index()

        # --- absolute datetimes (used for filtering traces) ---
        submitted_dt = parse_timestamp(raw.get("submitted_time"))

        mask = (
            (gpu_trace["machineId"].isin(machine_ids)) &
            (gpu_trace["time"] >= start_dt) &
            (gpu_trace["time"] <= end_dt)
        )
        job_gpu = gpu_trace.loc[mask].copy()

        print(f"  job_gpu shape after filtering: {job_gpu.shape}")
        if job_gpu.empty:
            print("  ⚠ No GPU rows matched this job")

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

                    cpu_cores_required=0,
                    gpu_units_required=gpus,
                    allocated_cpu_cores=0,
                    allocated_gpu_units=gpus,

                    end_state=status,
                    scheduled_nodes=scheduled_nodes,

                    cpu_trace=0,
                    gpu_trace=job_gpu,
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

    # Find max end timestamp across jobs
    #end_ts = max(j.end_time for j in jobs_list if j.end_time is not None)
    end_ts = 3600

    return WorkloadData(
        jobs=jobs_list,
        telemetry_start=0, telemetry_end=int(end_ts - start_ts),
        start_date=datetime.fromtimestamp(start_ts, timezone.utc),
    )
