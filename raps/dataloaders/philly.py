import os
import json
import csv
import datetime
import pandas as pd
import warnings
from raps.job import Job

DATE_FORMAT_STR = "%Y-%m-%d %H:%M:%S"

def parse_date(s):
    if not s or s == "None":
        return None
    # strip possible timezone labels like "PST"/"PDT"
    s = s.replace(" PST", "").replace(" PDT", "")
    return datetime.datetime.strptime(s, DATE_FORMAT_STR)

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

    # --- 2. CPU util ---
    cpu_file = os.path.join(trace_dir, "cluster_cpu_util")
    cpu_util = pd.read_csv(cpu_file)
    # cpu_util has columns: time, machine_id, cpu_util

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

    # --- 4. Job log ---
    job_file = os.path.join(trace_dir, "cluster_job_log")
    with open(job_file) as f:
        job_log = json.load(f)

    jobs = []
    for raw in job_log:
        jobid = raw.get("jobid")
        user = raw.get("user")
        status = raw.get("status")
        submitted = parse_date(raw.get("submitted_time"))

        attempts = raw.get("attempts", [])
        start, end = None, None
        if attempts:
            start = parse_date(attempts[0].get("start_time"))
            end = parse_date(attempts[-1].get("end_time"))

        wall_time = None
        if start and end:
            wall_time = (end - start).total_seconds()

        # Which machines did this job run on?
        machine_ids = []
        gpus = 0
        if attempts and "detail" in attempts[0]:
            for detail in attempts[0]["detail"]:
                mid = detail["ip"]
                machine_ids.append(mid)
                gpus += len(detail.get("gpus", []))

        # Collect utilization traces for each machine this job touched
        job_cpu = cpu_util[cpu_util["machine_id"].isin(machine_ids)]
        job_gpu = gpu_util[gpu_util["machineId"].isin(machine_ids)]

        job = Job(
            job_id=jobid,
            name=f"philly-{jobid}",
            user=user,
            nodes_required=len(machine_ids) if machine_ids else None,
            wall_time=wall_time,
            start_time=start,
            end_time=end,
            queue_time=submitted,
            scheduled_nodes=machine_ids,
            cpu_trace=job_cpu if not job_cpu.empty else None,
            gpu_trace=job_gpu if not job_gpu.empty else None,
            priority=None,
            end_state=status
        )
        jobs.append(job)

    return jobs
