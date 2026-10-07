"""
Dataloader for the public "Frontier Job-Centric Telemetry Dataset" (OLCF, DOI 10.13139/OLCF/3013979).

https://doi.org/10.13139/OLCF/3013979

This is a different layout from the private slurm/joblive + jobprofile data read by
`raps.dataloaders.frontier`: one scheduler table for all jobs plus a directory of per-job
power/temperature parquet files, partitioned by the date each job ended.

# To simulate one day (jobs are grouped by end date, so a day holds the jobs that ended on it)
DPATH=/store/datasets/hpc/frontier/telemetry
raps run --dataloader raps.dataloaders.frontier_jobcentric \
    -f $DPATH/frontier-completed-job-info.parquet,$DPATH/job-telemetry/2024-01-17

Note that the dataset is a filtered sample of Frontier's jobs (about 17% of eligible jobs, and only
COMPLETED/TIMEOUT/CANCELLED), so a replayed day under-fills the machine relative to reality, and the
first and last hours of a day are missing jobs that started or ended outside the sampled end date.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from ..job import job_dict, Job
from ..utils import power_to_utilization, WorkloadData
from .frontier import aging_boost, xname_to_index

# Columns of <job_idx>-cleaned-power.parquet used for the traces. These are gap-filled sums over all
# nodes of the job, binned at 1 second.
CPU_COL = 'cpu_agg_total_power'
GPU_COL = 'gpu_agg_total_power'
SOURCE_BIN_SECONDS = 1


def load_data(files, **kwargs):
    """
    Reads the job table and the telemetry for one end-date partition.

    Parameters
    ----------
    files : list[str]
        [path to frontier-completed-job-info.parquet, path to job-telemetry/<YYYY-MM-DD>]
    """
    assert len(files) == 2, \
        "Frontier job-centric dataloader requires two paths: the job-info parquet and a job-telemetry/<date> directory"
    jobs_df = pd.read_parquet(files[0], engine='pyarrow')
    telemetry_dir = Path(files[1])
    return load_data_from_df(jobs_df, telemetry_dir, **kwargs)


def _read_trace(telemetry_dir: Path, job_idx: str, quanta: int):
    """
    Returns (cpu_power, gpu_power) as arrays with one value per `quanta` seconds, or None if the
    job has no telemetry on disk. Each value is the mean of the 1 s bins in that interval.
    """
    path = telemetry_dir / job_idx / f'{job_idx}-cleaned-power.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=[CPU_COL, GPU_COL], engine='pyarrow')
    step = quanta // SOURCE_BIN_SECONDS
    out = []
    for col in (CPU_COL, GPU_COL):
        vals = df[col].to_numpy(dtype=float)
        pad = (-len(vals)) % step
        if pad:
            # Average the trailing partial interval over the bins it actually has.
            vals = np.concatenate([vals, np.full(pad, np.nan)])
        out.append(np.nanmean(vals.reshape(-1, step), axis=1))
    return out[0], out[1]


def load_data_from_df(jobs_df: pd.DataFrame, telemetry_dir: Path, **kwargs):
    """
    Builds RAPS jobs from the job table, restricted to the partition in `telemetry_dir`.

    Returns a WorkloadData whose time zero is the earliest job start in the partition.
    """
    config = kwargs.get('config')
    jid = kwargs.get('jid', '*')
    debug = kwargs.get('debug')
    quanta = config['TRACE_QUANTA']
    assert quanta % SOURCE_BIN_SECONDS == 0

    date_dir = telemetry_dir.name
    jobs_df = jobs_df[jobs_df['date_dir'] == date_dir]
    if jobs_df.empty:
        raise ValueError(f"No jobs with date_dir={date_dir} in the job-info table")
    jobs_df = jobs_df.sort_values('start_time').reset_index(drop=True)

    # Only some date directories are populated in partial copies of the dataset, so skip jobs
    # without telemetry rather than failing, and use the rest to define the time window.
    have = {p.name for p in telemetry_dir.iterdir() if p.is_dir()}
    n_missing = int((~jobs_df['job_idx'].isin(have)).sum())
    if n_missing:
        print(f"Frontier job-centric: {n_missing} of {len(jobs_df)} jobs in {date_dir} have no telemetry; skipping")
        jobs_df = jobs_df[jobs_df['job_idx'].isin(have)].reset_index(drop=True)

    telemetry_start_timestamp = jobs_df['start_time'].min()
    telemetry_end_timestamp = jobs_df['end_time'].max()
    telemetry_start = 0
    telemetry_end = int((telemetry_end_timestamp - telemetry_start_timestamp).total_seconds())
    if debug:
        print("num_jobs:", len(jobs_df), "window:", telemetry_start_timestamp, "->", telemetry_end_timestamp)

    def seconds(ts):
        return (ts - telemetry_start_timestamp).total_seconds()

    jobs = []
    for row in tqdm(jobs_df.itertuples(index=False), total=len(jobs_df), desc="Processing Jobs"):
        job_idx = row.job_idx
        if jid != '*' and jid != job_idx:
            continue

        trace = _read_trace(telemetry_dir, job_idx, quanta)
        if trace is None:
            continue
        cpu_power, gpu_power = trace
        if gpu_power.size == 0:
            continue

        nodes_required = int(row.node_count)
        cpu_min = nodes_required * config['POWER_CPU_IDLE'] * config['CPUS_PER_NODE']
        cpu_max = nodes_required * config['POWER_CPU_MAX'] * config['CPUS_PER_NODE']
        cpu_trace = power_to_utilization(cpu_power, cpu_min, cpu_max) * config['CPUS_PER_NODE']
        gpu_min = nodes_required * config['POWER_GPU_IDLE'] * config['GPUS_PER_NODE']
        gpu_max = nodes_required * config['POWER_GPU_MAX'] * config['GPUS_PER_NODE']
        gpu_trace = power_to_utilization(gpu_power, gpu_min, gpu_max) * config['GPUS_PER_NODE']
        cpu_trace[np.isnan(cpu_trace)] = 0
        gpu_trace[np.isnan(gpu_trace)] = 0

        start_time = seconds(row.start_time)
        end_time = seconds(row.end_time)
        submit_time = seconds(row.submit_time)
        expected_run_time = end_time - start_time
        trace_time = gpu_trace.size * quanta

        scheduled_nodes = [xname_to_index(x, config) for x in row.node_list]

        jobs.append(Job(job_dict(
            nodes_required=nodes_required,
            name=f"{row.project_id}/{job_idx}",
            account=row.user_id,
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace,
            nrx_trace=None,
            ntx_trace=None,
            end_state=row.completion_state,
            scheduled_nodes=scheduled_nodes,
            id=int(job_idx),
            priority=aging_boost(nodes_required),
            submit_time=submit_time,
            time_limit=int(row.wall_time) * 60,
            start_time=start_time,
            end_time=end_time,
            expected_run_time=expected_run_time,
            current_run_time=0,
            trace_time=trace_time,
            trace_start_time=0,
            trace_end_time=trace_time,
            trace_quanta=quanta,
            trace_missing_values=expected_run_time > trace_time,
        )))

    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start,
        telemetry_end=telemetry_end,
        start_date=telemetry_start_timestamp,
    )
