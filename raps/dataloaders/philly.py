"""
This is the dataloader for the Philly traces which is documented in this paper:

    Jeon, Myeongjae, et al. "Analysis of Large-Scale Multi-Tenant GPU clusters
    for DNN training workloads." 2019 USENIX Annual Technical Conference
    (USENIX ATC 19). 2019. https://www.usenix.org/system/files/atc19-jeon.pdf

Note on hardware specs:

    Philly only provides GPU memory sizes (12G & 24G) without clarifying GPU models.
    Hu et al. (2024) https://arxiv.org/html/2403.07648v1

    For estimating system power and FLOPS performance, we assume that the 2-GPU
    nodes used Tesla P100 (12 GB) GPUs and the 8-GPU nodes used Tesla P40 (24 GB)
    GPUs, consistent with hardware Microsoft deployed around 2017. Training is
    assumed to have been performed in 32-bit (FP32), and the CPUs are assumed
    to be 64-bit Intel Xeon E5-2690 v4.

The repository is available here:

    https://github.com/msr-fiddle/philly-traces

The data portion of the repo can be downloaded using one of the following methods:

    git clone https://github.com/msr-fiddle/philly-traces.git
    cd philly-traces
    git lfs pull

    wget https://github.com/msr-fiddle/philly-traces/raw/master/trace-data.tar.gz

    curl -L -o trace-data.tar.gz \
            https://github.com/msr-fiddle/philly-traces/raw/master/trace-data.tar.gz

After the file is downloaded, assuming its in /opt/data/philly/trace-data directory:

    /opt/data/philly/trace-data/trace-data.tar.gz

    cd /opt/data/philly/trace-data

    run `tar xvfz trace-data.tar.gz` which will unpack the following files:

        cluster_cpu_util     1.5G
        cluster_gpu_util     2.8G
        cluster_mem_util     2.2G
        cluster_job_log      37M
        cluster_machine_list 8K

    then run the following:

        python /path/to/raps/scripts/parse_philly_traces.py cluster_cpu_util
        python /path/to/raps/scripts/parse_philly_traces.py cluster_gpu_util

    this will parse these two files into two directories, cpu_by_day and gpu_by_day,
    creating one file for each day and adding the lines for that day into the files.

    sanity checks:

        wc -l cluster_cpu_util
         45028261 cluster_cpu_util
        wc -l cpu_by_day/*.csv
         45350898 total

        wc -l cluster_gpu_util
         44750641 cluster_gpu_util
        wc -l gpu_by_day/*.csv
         44750640 total

Running a replay simulation:

    python main.py run-parts -x philly -f /opt/data/philly/trace-data \
            --start 2017-10-03T00:00 --end 2017-10-04T00:00

Once the dataloader has been run at least once, it will dump npz files into a directory,
so they can be replayed again without having to go through the expensive extractoin process,
using e.g.:

    python main.py run-parts -x philly -f raps-output-5efefa3

Note: it is possible to run simulations for an user-defined length of time between
10/3/2017 to 12/15/2017.

"""

import csv
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from tqdm import tqdm

from raps.job import Job, job_dict
from raps.utils import WorkloadData

DATE_FORMAT_STR = "%Y-%m-%d %H:%M:%S"
DEFAULT_START = "2017-10-03T00:00"
DEFAULT_END = "2017-10-04T00:00"


def to_epoch(ts_str):
    """Convert a timestamp string or int/float into epoch seconds."""
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
    traces = {}
    current = start_dt.date()

    while current <= end_dt.date():
        daily_file = os.path.join(trace_dir, f"{current}.csv")
        if os.path.exists(daily_file):
            df = pd.read_csv(
                daily_file,
                names=["time", "machineId", colname],  # no header in daily CSVs
                dtype={"machineId": str, colname: str},  # avoid DtypeWarning
            )

            # Normalize time column (strip PST/PDT, parse datetime)
            df["time"] = df["time"].str.replace(" PST", "").str.replace(" PDT", "")
            df["time"] = pd.to_datetime(
                df["time"], errors="coerce", format=DATE_FORMAT_STR
            )

            # Convert util column to numeric (NA/invalid → NaN)
            df[colname] = pd.to_numeric(df[colname], errors="coerce")

            traces[current] = df
        else:
            print(f"⚠ No trace file for {current}")
        current += timedelta(days=1)

    if not traces:
        return {}

    return traces


def parse_date(s):
    """Parse a Philly trace date string into a datetime object."""
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
    debug = kwargs.get("debug")
    print("started reading of philly traces... please be patient...", flush=True)

    # extract --start from kwargs
    start_ts = to_epoch(kwargs.get("start", DEFAULT_START))
    end_ts = to_epoch(kwargs.get("end", DEFAULT_END))

    assert len(files) == 1, "Expecting a single directory path"
    trace_dir = files[0]
    gpu_trace_dir = os.path.join(files[0], "gpu_by_day")
    config = kwargs.get("config")
    gpus_per_node = config.get("GPUS_PER_NODE")
    if gpus_per_node is None:
        raise ValueError("Must pass gpus_per_node (2 or 8)")

    # --- 1. Machine list ---
    machine_file = os.path.join(trace_dir, "cluster_machine_list")
    machines = {}
    with open(machine_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["machineId"]
            machines[mid] = {
                "num_gpus": int(row[" number of GPUs"]),
                "gpu_mem": row[" single GPU mem"].strip(),
            }

    partition_machines = {
        mid: info for mid, info in machines.items() if info["num_gpus"] == gpus_per_node
    }

    # Build node → index mapping for this partition
    node_mapping = {
        mid: idx for idx, mid in enumerate(sorted(partition_machines.keys()))
    }

    # Assign partition ID (e.g. 0 for 2-GPU, 1 for 8-GPU)
    partition_id = 0 if gpus_per_node == 2 else 1

    # --- 3. GPU util ---
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)

    # --- 4. Job log ---
    job_file = os.path.join(trace_dir, "cluster_job_log")
    with open(job_file, encoding="utf-8") as f:
        job_log = json.load(f)

    # --- First pass: filter jobs by date range ---
    filtered_log = []
    for raw in job_log:
        submitted = raw.get("submitted_time")
        if submitted is None or submitted == "None":
            continue
        if isinstance(submitted, (int, float)):
            submitted_dt = datetime.fromtimestamp(int(submitted))
        else:
            submitted_dt = parse_date(submitted)
        if submitted_dt and start_dt <= submitted_dt <= end_dt:
            filtered_log.append(raw)
    job_log = filtered_log

    # Filter job_log to only jobs matching the partition's gpus_per_node
    if gpus_per_node is not None:
        filtered_log = []
        for raw in job_log:
            attempts = raw.get("attempts", [])
            if attempts and "detail" in attempts[0]:
                # Count GPUs from the first detail
                gpus = sum(
                    len(detail.get("gpus", [])) for detail in attempts[0]["detail"]
                )
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

    # --- Pre-load all traces for the given date range ---
    cpu_trace_dir = os.path.join(trace_dir, "cpu_by_day")
    gpu_trace_dir = os.path.join(trace_dir, "gpu_by_day")
    all_cpu_traces = load_traces_by_day(cpu_trace_dir, start_dt, end_dt, "cpu_util")
    all_gpu_traces = load_traces_by_day(gpu_trace_dir, start_dt, end_dt, "gpu_util")

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
        if num_nodes == 0:
            continue
        gpus_per_node = gpus // num_nodes

        # --- absolute datetimes (used for filtering traces) ---
        submitted_dt = parse_timestamp(raw.get("submitted_time"))

        job_start = start
        job_end = end

        if not job_start or not job_end:
            continue

        # --- CPU utilization traces ---
        cpu_dfs = []
        current_date = job_start.date()
        while current_date <= job_end.date():
            if current_date in all_cpu_traces:
                cpu_dfs.append(all_cpu_traces[current_date])
            current_date += timedelta(days=1)

        if not cpu_dfs:
            job_cpu_trace = []
        else:
            job_cpu_df = pd.concat(cpu_dfs, ignore_index=True)
            mask = (
                (job_cpu_df["machineId"].isin(machine_ids))
                & (job_cpu_df["time"] >= start)
                & (job_cpu_df["time"] <= end)
            )
            job_cpu = job_cpu_df.loc[mask].copy()

            if len(machine_ids) > 1:
                job_cpu = job_cpu.groupby("time")["cpu_util"].mean().reset_index()

            job_cpu_trace = (job_cpu["cpu_util"].to_numpy() * 0.01).tolist()

        # --- GPU utilization traces ---
        gpu_dfs = []
        current_date = job_start.date()
        while current_date <= job_end.date():
            if current_date in all_gpu_traces:
                gpu_dfs.append(all_gpu_traces[current_date])
            current_date += timedelta(days=1)

        if not gpu_dfs:
            job_gpu_trace = []
        else:
            job_gpu_df = pd.concat(gpu_dfs, ignore_index=True)
            mask = (
                (job_gpu_df["machineId"].isin(machine_ids))
                & (job_gpu_df["time"] >= start)
                & (job_gpu_df["time"] <= end)
            )
            job_gpu = job_gpu_df.loc[mask].copy()

            if len(machine_ids) > 1:
                job_gpu = job_gpu.groupby("time")["gpu_util"].mean().reset_index()

            job_gpu_trace = (
                job_gpu["gpu_util"].to_numpy() * 0.01 * gpus_per_node
            ).tolist()

        if machine_ids:
            submit_time = submitted.timestamp()
            start_time = start.timestamp()
            end_time = end.timestamp()

            if not submit_time or not start_time or not end_time:
                tqdm.write(
                    f"skipped {jobid} b/c missing submit_time, start_time, or end_time"
                )
                continue

            scheduled_nodes = [
                node_mapping[mid] for mid in machine_ids if mid in node_mapping
            ]

            if submit_time and start_time and end_time:

                job = job_dict(
                    id=jobid,
                    name=f"philly-{jobid}",
                    account=user if user else "unknown",
                    nodes_required=len(machine_ids),
                    partition=partition_id,
                    cpu_cores_required=1,
                    gpu_units_required=gpus_per_node,
                    end_state=status,
                    scheduled_nodes=scheduled_nodes,
                    cpu_trace=job_cpu_trace,
                    gpu_trace=job_gpu_trace,
                    ntx_trace=[],
                    nrx_trace=[],
                    submit_time=submit_time,
                    start_time=start_time,
                    end_time=end_time,
                    time_limit=end_time,
                    expected_run_time=wall_time if wall_time else 0,
                    trace_start_time=start_time,  # None,
                    trace_end_time=end_time,  # None,
                    trace_quanta=60,
                    trace_missing_values=False
                )
                if job_cpu_trace and job_gpu_trace:
                    jobs_list.append(Job(job))
                else:
                    tqdm.write(f"skipping {job['id']} b/c either no cpu or gpu trace")

            if debug:
                tqdm.write(f"{job['id']} start: {job['start_time']} end: {job['end_time']}")

    return WorkloadData(
        jobs=jobs_list,
        telemetry_start=start_ts,
        telemetry_end=end_ts,
        start_date=datetime.fromtimestamp(start_ts, timezone.utc),
    )
