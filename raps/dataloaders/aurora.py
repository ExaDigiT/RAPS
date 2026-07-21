import pandas as pd
from pathlib import Path
from raps.telemetry import Job, job_dict
from raps.utils import WorkloadData
from datetime import datetime, timezone

""" 
Download DIM_JOB_COMPOSITE dataset from https://reports.alcf.anl.gov/data/aurora.html 

Test case:

    raps run -f /opt/data/aurora/ANL-ALCF-DJC-AURORA_20250127_20251031.csv \
                --system aurora --policy fcfs --arrival poisson

Note, that currently only reading NJOBS from the csv due to time constraints.
Increase NJOBS to 663507 to read all jobs.

"""
NJOBS = 1000

def load_data(local_dataset_path, **kwargs):
    """
    Aurora dataloader.
    """
    if isinstance(local_dataset_path, list):
        filepath = Path(local_dataset_path[0])
    else:
        filepath = Path(local_dataset_path)

    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Reading data from {filepath}")

    jobs = []
    chunksize = 10000
    
    required_columns = [
        "COBALT_JOBID", "JOB_NAME", "QUEUED_TIMESTAMP", "START_TIMESTAMP", "END_TIMESTAMP",
        "NODES_REQUESTED", "NODES_USED", "CORES_REQUESTED", "CORES_USED",
        "WALLTIME_SECONDS", "RUNTIME_SECONDS", "USERNAME_GENID", "LOCATION"
    ]

    for chunk in pd.read_csv(filepath, chunksize=chunksize, on_bad_lines='warn', nrows=NJOBS):
        # Drop rows where essential timestamp data is missing
        chunk.dropna(subset=['QUEUED_TIMESTAMP', 'START_TIMESTAMP', 'END_TIMESTAMP'], inplace=True)

        for _, row in chunk.iterrows():
            submit_time = int(pd.to_datetime(row["QUEUED_TIMESTAMP"]).timestamp())
            start_time = int(pd.to_datetime(row["START_TIMESTAMP"]).timestamp())
            end_time = int(pd.to_datetime(row["END_TIMESTAMP"]).timestamp())
            job_name = row.get("JOB_NAME", "N/A")
            job_id = job_name.split('.')[0]

            job = job_dict(
                id=job_id,
                name=job_name,
                submit_time=submit_time,
                start_time=start_time,
                end_time=end_time,
                time_limit=int(row.get("WALLTIME_SECONDS", 0)),
                expected_run_time=int(row.get("RUNTIME_SECONDS", 0)),
                nodes_required=int(row.get("NODES_REQUESTED", 0)),
                cpu_cores_required=int(row.get("CORES_REQUESTED", 0)),
                account=str(row.get("USERNAME_GENID", "N/A")),
                #scheduled_nodes=str(row.get("LOCATION", "")).split(','),
                scheduled_nodes=[], #str(row.get("LOCATION", "")),
                # The following are placeholders as they are not in the CSV
                gpu_trace=0,
                cpu_trace=0,
                nrx_trace=[],
                ntx_trace=[],
                end_state="COMPLETED",
                priority=0,
                current_run_time=0,
                trace_time=submit_time,
                trace_start_time=start_time,
                trace_end_time=end_time,
                trace_quanta=1,
            )
            jobs.append(Job(job))

    if not jobs:
        return WorkloadData(jobs=[], telemetry_start=0, telemetry_end=0, start_date=datetime.now(timezone.utc))

    # Normalize times so first start = 0
    t0 = min((j.start_time for j in jobs), default=0)
    for j in jobs:
        j.submit_time -= t0
        j.start_time -= t0
        j.end_time -= t0
        j.trace_time -= t0
        j.trace_start_time -= t0
        j.trace_end_time -= t0

    telemetry_start = 0
    telemetry_end = max((j.end_time for j in jobs), default=0)
    start_date = datetime.fromtimestamp(t0, timezone.utc)

    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start,
        telemetry_end=telemetry_end,
        start_date=start_date,
    )
