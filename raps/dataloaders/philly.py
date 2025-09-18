import os
import json
import csv
import pandas as pd
import warnings

from datetime import datetime, timezone
from tqdm import tqdm
from raps.job import job_dict, Job
from raps.utils import WorkloadData

DATE_FORMAT_STR = "%Y-%m-%d %H:%M:%S"

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
    assert len(files) == 1, "Expecting a single directory path"
    trace_dir = files[0]

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

    # build node → index mapping
    node_mapping = {mid: idx for idx, mid in enumerate(sorted(machines.keys()))}

    # --- 2. CPU util ---
    cpu_file = os.path.join(trace_dir, "cluster_cpu_util")
    cpu_util = pd.read_csv(cpu_file)
    # cpu_util has columns: time, machine_id, cpu_util
    cpu_util["time"] = cpu_util["time"].str.replace(" PST","").str.replace(" PDT","")
    #cpu_util["time"] = pd.to_datetime(cpu_util["time"], format="%Y-%m-%d %H:%M:%S")
    # now cpu_util has: time (datetime), machine_id, cpu_util
    cpu_util["time"] = cpu_util["time"].apply(parse_date)

    # --- 3. GPU util ---
    gpu_file = os.path.join(trace_dir, "cluster_gpu_util")

    with warnings.catch_warnings(record=True) as wlist:
        gpu_util = pd.read_csv(
            gpu_file,
            engine="python",
            on_bad_lines="skip"
        )
        if wlist:
            warnings.warn(
                f"cluster_gpu_util: skipped {len(wlist)} malformed lines while reading {gpu_file}",
                UserWarning
            )

    # Convert time to datetime
    gpu_util["time"] = pd.to_datetime(gpu_util["time"], errors="coerce").dt.tz_localize(None)

    # Identify GPU columns
    gpu_cols = [c for c in gpu_util.columns if c.startswith("gpu")]

    # Collapse per row: sum all GPU utilizations and divide by 100
    gpu_util["gpu_util"] = gpu_util[gpu_cols].sum(axis=1) / 100.0

    # Keep only collapsed util plus metadata
    gpu_util = gpu_util[["time", "machineId", "gpu_util"]]

    print("Sample GPU util after preprocess:", gpu_util.head())

    # --- 4. Job log ---
    job_file = os.path.join(trace_dir, "cluster_job_log")
    with open(job_file) as f:
        job_log = json.load(f)

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

        print("Job", jobid)
        print("machine_ids from job:", machine_ids[:5])
        print("gpu_util machineId sample:", gpu_util["machineId"].unique()[:5])
        print("start, end:", start, end)
        print("gpu_util time range:", gpu_util["time"].min(), gpu_util["time"].max())

        # GPU utilization traces
        job_gpu = None
        if machine_ids and start and end:
            mask = (
                gpu_util["machineId"].isin(machine_ids) &
                (gpu_util["time"] >= start) &
                (gpu_util["time"] <= end)
            )
            job_gpu = gpu_util.loc[mask].copy()

            # Aggregate across machines if >1 machine
            if len(machine_ids) > 1:
                job_gpu = job_gpu.groupby("time")["gpu_util"].sum().reset_index()

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
                    partition=0,
                    priority=0,

                    cpu_cores_required=0,
                    gpu_units_required=gpus,
                    allocated_cpu_cores=0,
                    allocated_gpu_units=gpus,

                    end_state=status,
                    scheduled_nodes=scheduled_nodes,

                    cpu_trace=job_cpu,
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
    end_ts = max(j.end_time for j in jobs_list if j.end_time is not None)

    return WorkloadData(
        jobs=jobs_list,
        telemetry_start=0, telemetry_end=int(end_ts - start_ts),
        start_date=datetime.fromtimestamp(start_ts, timezone.utc),
    )
