import numpy as np
import os
import re
import pandas as pd

from datetime import datetime
from scipy.sparse import csr_matrix as csr
from tqdm import tqdm

DEFAULT_START = "2021-05-21T00:00"
DEFAULT_END = "2021-05-22T00:00"


def to_epoch(s: str) -> int:
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except ValueError:
        return int(datetime.strptime(s, "%d%m%Y").timestamp())


def parse_dt(s: str) -> datetime:
    try:
        # handles 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM[:SS]'
        return datetime.fromisoformat(s)
    except ValueError:
        # legacy support for DDMMYYYY → midnight
        return datetime.strptime(s, "%d%m%Y")


def load_slurm_log(slurm_path: str, start_date: str, end_date: str):
    """
    Load Slurm log and filter jobs by submission window.

    Args:
        slurm_path: Path to local slurm-log.csv
        start_date: "DDMMYYYY" inclusive start
        end_date:   "DDMMYYYY" exclusive end

    Returns:
        tuple(
            pandas.DataFrame filtered on date window,
            set of CPU-only job IDs,
            set of GPU-using job IDs
        )
    """
    df = pd.read_csv(slurm_path)
    # Convert submit times
    df['time_submit'] = pd.to_datetime(df['time_submit'], unit='s')
    dt0 = parse_dt(start_date)
    dt1 = parse_dt(end_date)
    window = df[(df['time_submit'] >= dt0) & (df['time_submit'] < dt1)]

    # Detect GPU jobs via gres_used or tres_alloc
    gres = window['gres_used'].fillna("").astype(str)
    tres = window['tres_alloc'].fillna("").astype(str)
    gpu_jobs = set(
        window.loc[
            gres.str.contains("gpu", case=False) |
            tres.str.contains(r"(?:1001|1002)=", regex=True),
            'id_job'
        ]
    )
    cpu_jobs = set(window['id_job']) - gpu_jobs
    return window, cpu_jobs, gpu_jobs


def build_or_load_manifest(s3, bucket: str, prefix: str, manifest_path: str):
    """
    Build a one-time manifest of all .csv keys under cpu/ and gpu/ in S3,
    or load an existing manifest from disk.

    Args:
        s3: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 dataset root prefix (e.g. "datacenter-challenge/202201/")
        manifest_path: local path to cache the manifest

    Returns:
        List[str]: all S3 keys ending in .csv under cpu/ and gpu/
    """
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return [line.strip() for line in f]

    # Otherwise build manifest
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    total_pages = {'cpu': 791, 'gpu': 110}
    progress = tqdm(total=sum(total_pages.values()), desc="Building file-manifest.txt", unit="page")

    for kind in ('cpu', 'gpu'):
        pfx = prefix + f"{kind}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=pfx):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.lower().endswith('.csv'):
                    keys.append(key)
            progress.update(1)

    progress.close()

    # Cache on disk
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        for key in keys:
            f.write(key + '\n')
    return keys


def filter_keys_by_jobs(all_keys: list, job_ids: set):
    """
    Filter a list of S3 keys to those belonging to specified job IDs.

    Args:
        all_keys: list of S3 keys from manifest
        job_ids:  set of job IDs (int)

    Returns:
        List[str] of keys matching CPU or GPU jobs
    """
    selected = []
    gpu_pattern = re.compile(r'-r(\d+)-')
    for key in all_keys:
        # CPU keys: prefix/jobid-...-timeseries.csv or -summary.csv
        if '/cpu/' in key:
            fname = os.path.basename(key)
            parts = fname.split('-', 1)
            try:
                jid = int(parts[0])
            except ValueError:
                continue
            if jid in job_ids:
                selected.append(key)
        # GPU keys: detect -r<jobid>- in filename
        elif '/gpu/' in key:
            fname = os.path.basename(key)
            m = gpu_pattern.search(fname)
            if m and int(m.group(1)) in job_ids:
                selected.append(key)
    return selected


def proc_cpu_series(dfi):
    dfi = dfi[~dfi.Step.isin([-1, -4, '-1', '-4'])].copy()
    dfi['CPUUtilization'] = dfi['CPUUtilization'].fillna(0) / 100.0

    t = pd.to_datetime(dfi.EpochTime, unit='s')
    start_time = t.min()
    dfi['t'] = ((t - start_time).dt.total_seconds() // 10).astype(int)
    dfi['sid'] = pd.factorize(dfi.Step)[0]

    useries = dfi.Series.unique()
    inds = np.arange(dfi.t.max() + 1)
    df = pd.DataFrame({'t': inds})
    Xm, Xrss, Xvm, Xreadmb, Xwritemb = (np.zeros((len(useries), len(inds))) for _ in range(5))

    for cnt, i in enumerate(useries):
        sift = dfi.Series == i
        M, N = len(inds), dfi.sid[sift].max() + 1

        for metric, arr, name in zip(
            ['CPUUtilization', 'RSS', 'VMSize', 'ReadMB', 'WriteMB'],
            [Xm, Xrss, Xvm, Xreadmb, Xwritemb],
            ['cpu', 'rss', 'vm', 'readmb', 'writemb']
        ):
            X = csr((dfi.loc[sift, metric], (dfi.loc[sift, 't'], dfi.loc[sift, 'sid'])), shape=(M, N))
            mm = np.array(X.max(axis=1).todense()).reshape(-1,)
            df[f'{name}_{i}'] = mm
            arr[cnt, :] = mm

    df['cpu_utilisation'] = Xm.mean(axis=0)
    df['rss'] = Xrss.sum(axis=0)
    df['vm'] = Xvm.sum(axis=0)
    df['readmb'] = Xreadmb.sum(axis=0)
    df['writemb'] = Xwritemb.sum(axis=0)
    df['timestamp'] = start_time + pd.to_timedelta(df.t * 10, unit='s')
    df['utime'] = df['timestamp'].astype('int64') // 10**9

    return df


def proc_gpu_series(cpu_df, dfi, gpu_cnt):
    # 1) Build CPU time range
    t_cpu_start = int(cpu_df.utime.min())
    t_cpu_end = int(cpu_df.utime.max())
    t_cpu = np.array([t_cpu_start, t_cpu_end, t_cpu_end - t_cpu_start])

    # 2) Safely convert the GPU timestamps to integer seconds
    #    (this handles strings like "1621607266.426")
    ts = pd.to_numeric(dfi["timestamp"], errors="coerce")  # float64 or NaN
    ts_int = ts.ffill().astype(float).astype(int)
    t0, t1 = ts_int.min(), ts_int.max()
    t_gpu = np.array([t0, t1, t1 - t0])

    # 3) Sanity‐check the durations match within 10%
    per_diff = ((t_cpu[1] - t_cpu[0]) - (t_gpu[1] - t_gpu[0])) / (t_gpu[1] - t_gpu[0]) * 100
    if abs(per_diff) > 10:
        # warn and proceed — GPU trace may be trimmed or misaligned
        tqdm.write(f"Warning: GPU‐CPU time mismatch {per_diff:.1f}% exceeds 10%; continuing anyway")

    # 4) Align GPU times onto CPU utime grid
    #    Use our integer‐second Series rather than the raw column
    dfi["t_fixed"] = ts_int - ts_int.min() + t_cpu_start

    # 5) Prepare output DataFrame with a utime column
    # ugpus = dfi.gpu_index.unique()
    gpu_df = pd.DataFrame({"utime": cpu_df["utime"].values})

    # 6) Interpolate each GPU field onto the CPU utime grid
    fields = [
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "memory_free_MiB",
        "memory_used_MiB",
        "temperature_gpu",
        "temperature_memory",
        "power_draw_W",
    ]
    for field in fields:
        # grab the float‐converted timestamp and the metric
        x1 = ts_int.values
        y1 = dfi[field].astype(float).values
        xv = cpu_df["utime"].values
        # numpy interpolation
        gpu_df[field] = np.interp(xv, x1, y1)

    # 7) Rename the GPU pct, memory pct, and power columns with the device index
    ren = {
        "gpu_index":            f"gpu_index_{gpu_cnt}",
        "utilization_gpu_pct":  f"gpu_util_{gpu_cnt}",
        "utilization_memory_pct": f"gpu_mempct_{gpu_cnt}",
        "memory_free_MiB":      f"gpu_memfree_{gpu_cnt}",
        "memory_used_MiB":      f"gpu_memused_{gpu_cnt}",
        "temperature_gpu":      f"gpu_temp_{gpu_cnt}",
        "temperature_memory":   f"gpu_memtemp_{gpu_cnt}",
        "power_draw_W":         f"gpu_power_{gpu_cnt}",
    }
    gpu_df.rename(columns=ren, inplace=True)

    return gpu_df, gpu_cnt + 1


def validate_job_traces(job, granularity=1):
    print(job)
    assert job.cpu_trace is not None, f"job {job.id} missing cpu_trace"
    assert job.gpu_trace is not None, f"job {job.id} missing gpu_trace"
    assert all(p >= 0 for p in job.cpu_trace), f"neg cpu power in job {job.id}"
    assert all(p >= 0 for p in job.gpu_trace), f"neg gpu power in job {job.id}"
    # Length sanity: at least wall_time/granularity samples
    needed = max(1, int(job.wall_time / granularity))
    assert len(job.cpu_trace) >= needed, f"cpu_trace too short for job {job.id}"
    assert len(job.gpu_trace) >= needed, f"gpu_trace too short for job {job.id}"
