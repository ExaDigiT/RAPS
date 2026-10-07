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

# The telemetry was measured on Frontier, so power is converted to utilization with Frontier's
# per-node ranges (config/frontier.yaml), independent of the system being simulated. The
# utilization is then scaled by the target system's CPUs/GPUs per node.
SRC_CPUS_PER_NODE, SRC_GPUS_PER_NODE = 1, 4
SRC_POWER_CPU_IDLE, SRC_POWER_CPU_MAX = 90, 280
SRC_POWER_GPU_IDLE, SRC_POWER_GPU_MAX = 88, 560
SRC_AVAILABLE_NODES = 9472


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

    # Optional row filter from the sim config's `filter`, a pandas query over the job table's
    # columns plus two derived ones:
    #   power_per_node  mean_power / node_count [W], a proxy for GPU intensity (idle nodes are
    #                   about 600 W, GPU-heavy jobs 1500 W and up)
    #   rand            reproducible uniform [0, 1) per job (seeded by `seed`), for random thinning
    # e.g. filter: "power_per_node > 1500"  or  filter: "rand < 0.5 and node_count >= 2"
    # The table has no AI/ML label (gpu_enabled is null before 2024-10 and project prefixes are
    # anonymized), so power_per_node is the closest proxy for GPU-heavy work.
    filter_str = kwargs.get('filter')
    if filter_str:
        rng = np.random.default_rng(kwargs.get('seed') or 0)
        n_before = len(jobs_df)
        jobs_df = jobs_df.assign(power_per_node=jobs_df['mean_power'] / jobs_df['node_count'],
                                 rand=rng.random(n_before))
        jobs_df = jobs_df.query(filter_str)
        print(f"Frontier job-centric: filter '{filter_str}' kept {len(jobs_df)} of {n_before} jobs")
        if jobs_df.empty:
            raise ValueError(f"filter '{filter_str}' removed every job in {date_dir}")

    # On a smaller target system (e.g. Lux, 504 nodes) drop jobs that cannot fit, and let the
    # scheduler place the rest instead of replaying Frontier's node ids.
    smaller_target = config['AVAILABLE_NODES'] < SRC_AVAILABLE_NODES
    if smaller_target:
        too_big = jobs_df['node_count'] > config['AVAILABLE_NODES']
        print(f"Frontier job-centric: dropping {int(too_big.sum())} of {len(jobs_df)} jobs larger than "
              f"{config['AVAILABLE_NODES']} nodes")
        jobs_df = jobs_df[~too_big]
    jobs_df = jobs_df.sort_values('start_time').reset_index(drop=True)

    # Partial copies of the dataset can have job folders without their files (or no folders at
    # all), so skip jobs with no power file rather than failing, and report how many.
    has_power = np.array([(telemetry_dir / j / f'{j}-cleaned-power.parquet').exists() for j in jobs_df['job_idx']])
    n_missing = int((~has_power).sum())
    if n_missing:
        print(f"Frontier job-centric: {n_missing} of {len(jobs_df)} jobs in {date_dir} have no "
              f"cleaned-power file; skipping")
        jobs_df = jobs_df[has_power].reset_index(drop=True)
    if jobs_df.empty:
        raise ValueError(f"No telemetry files found for {date_dir} under {telemetry_dir}")

    # Time zero is floored to a multiple of the trace quanta (on the epoch grid): the engine only
    # ticks at timesteps divisible by time_delta, so with time_delta == quanta a start offset that
    # is not a multiple of it would never tick.
    telemetry_start_timestamp = jobs_df['start_time'].min().floor(f'{quanta}s')
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
        cpu_min = nodes_required * SRC_POWER_CPU_IDLE * SRC_CPUS_PER_NODE
        cpu_max = nodes_required * SRC_POWER_CPU_MAX * SRC_CPUS_PER_NODE
        cpu_trace = power_to_utilization(cpu_power, cpu_min, cpu_max) * config['CPUS_PER_NODE']
        gpu_min = nodes_required * SRC_POWER_GPU_IDLE * SRC_GPUS_PER_NODE
        gpu_max = nodes_required * SRC_POWER_GPU_MAX * SRC_GPUS_PER_NODE
        gpu_trace = power_to_utilization(gpu_power, gpu_min, gpu_max) * config['GPUS_PER_NODE']
        if smaller_target:
            # Below Frontier's configured idle (negative utilization) reproduces Frontier's own
            # measured power, but means nothing relative to another system's idle and max.
            cpu_trace = np.clip(cpu_trace, 0, config['CPUS_PER_NODE'])
            gpu_trace = np.clip(gpu_trace, 0, config['GPUS_PER_NODE'])
        cpu_trace[np.isnan(cpu_trace)] = 0
        gpu_trace[np.isnan(gpu_trace)] = 0

        start_time = seconds(row.start_time)
        end_time = seconds(row.end_time)
        submit_time = seconds(row.submit_time)
        expected_run_time = end_time - start_time
        trace_time = gpu_trace.size * quanta

        # Frontier xnames only mean something on a Frontier-sized layout; on a smaller system
        # leave placement to the scheduler.
        scheduled_nodes = None if smaller_target else [xname_to_index(x, config) for x in row.node_list]

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
            # Slurm lets TIMEOUT jobs run a little past their limit; RAPS raises if a rescheduled
            # job does, so make the limit cover the recorded runtime.
            time_limit=max(int(row.wall_time) * 60, int(np.ceil(expected_run_time / quanta)) * quanta),
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
