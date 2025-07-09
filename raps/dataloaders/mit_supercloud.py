#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT Supercloud job trace processing module with load_data function.
"""

import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix as csr
from tqdm import tqdm

from raps.job import job_dict


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
    t_cpu_end   = int(cpu_df.utime.max())
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
        print(f"Warning: GPU‐CPU time mismatch {per_diff:.1f}% exceeds 10%; continuing anyway")

    # 4) Align GPU times onto CPU utime grid
    #    Use our integer‐second Series rather than the raw column
    dfi["t_fixed"] = ts_int - ts_int.min() + t_cpu_start

    # 5) Prepare output DataFrame with a utime column
    ugpus = dfi.gpu_index.unique()
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
        "utilization_memory_pct":f"gpu_mempct_{gpu_cnt}",
        "memory_free_MiB":      f"gpu_memfree_{gpu_cnt}",
        "memory_used_MiB":      f"gpu_memused_{gpu_cnt}",
        "temperature_gpu":      f"gpu_temp_{gpu_cnt}",
        "temperature_memory":   f"gpu_memtemp_{gpu_cnt}",
        "power_draw_W":         f"gpu_power_{gpu_cnt}",
    }
    gpu_df.rename(columns=ren, inplace=True)

    return gpu_df, gpu_cnt + 1

def load_data(local_dataset_path, **kwargs):
    """
    Load MIT Supercloud job traces.
    Expects:
      local_dataset_path/
        metadata/
          file_list.csv
          job_user_date_full.csv
        202201/            # hard-coded for now; change as needed
          cpu/...-timeseries.csv
          gpu/...-timeseries.csv
          slurm-log.csv
    Returns:
      jobs_list, sim_start_time, sim_end_time
    """
    # 1) Unpack list if necessary
    if isinstance(local_dataset_path, list):
        if len(local_dataset_path) != 1:
            raise ValueError("MIT Supercloud loader accepts exactly one path")
        local_dataset_path = local_dataset_path[0]

    # 2) Read metadata
    meta_dir = os.path.join(local_dataset_path, "metadata")
    file_list_df = pd.read_csv(os.path.join(meta_dir, "file_list.csv"), sep="\t")
    job_index_df = pd.read_csv(os.path.join(meta_dir, "job_user_date_full.csv"))

    # 3) Date filtering settings
    start_date_str = kwargs.get("start_date", "21052021")
    end_date_str   = kwargs.get("end_date",   "22052021")
    jid            = kwargs.get("jid",          "*")
    # determine whether this is the CPU or GPU partition
    part = kwargs.get("partition", "").lower()
    cpu_only = ("cpu" in part) and ("gpu" not in part)
    gpu_only = ("gpu" in part) and ("cpu" not in part)

    start_ts = int(datetime.strptime(start_date_str, "%d%m%Y").timestamp())
    end_ts   = int(datetime.strptime(end_date_str,   "%d%m%Y").timestamp())
    requested_duration = end_ts - start_ts

    # 4) Select jobs in time window
    selected_df = job_index_df[
        (job_index_df.start > start_ts) &
        (job_index_df.start < end_ts)
    ].copy()

######
    data_subdir = "202201"  # hard-coded folder name
    print(local_dataset_path, data_subdir)

    # --- 1) Load and filter Slurm log for GPU jobs in [start_ts, end_ts) ---
    slurm_path = os.path.join(local_dataset_path, data_subdir, "slurm-log.csv")
    slurm_df   = pd.read_csv(slurm_path)

    # Keep only rows within your date window
    sl = slurm_df[
        (slurm_df.time_submit >= start_ts) &
        (slurm_df.time_submit <  end_ts)
    ]

    # Filter to those that actually used GPUs
    def row_uses_gpu(r):
        return ("gpu" in str(r.get("gres_used","")).lower()
                or "1001=" in str(r.get("tres_alloc",""))
                or "1002=" in str(r.get("tres_alloc","")))
    gpu_sl = sl[sl.apply(row_uses_gpu, axis=1)]

    gpu_job_ids = set(gpu_sl.id_job.unique())
    print(f"→ Found {len(gpu_job_ids)} GPU‐using jobs in your date range")

    # --- 2) Pull their GPU timeseries paths from file_list.csv ---
    #gpu_entries = file_list_df[
    #    file_list_df["File Name"].str.contains("/gpu/")
    #].copy()

    # should match both "gpu/..." at the start _and_ anywhere else
    #gpu_entries = file_list_df[
    #    file_list_df["File Name"].str.contains(r"(^|/)gpu/")
    #].copy()

    # Option 2: simple substring match (matches anywhere “gpu/” appears)
    gpu_entries = file_list_df[
        file_list_df["File Name"].str.contains("gpu/")
    ].copy()

    gpu_entries["job_id"] = (
        gpu_entries["File Name"]
          .str.extract(r"/(\d+)-", expand=False)
          .astype(int)
    )
    gpu_sel = gpu_entries[gpu_entries["job_id"].isin(gpu_job_ids)]
    gpu_files = gpu_sel["File Name"].tolist()
    print(f"→ Will process {len(gpu_files)} GPU files")

    # --- 3) Combine with your CPU list and dedupe ---
    cpu_files = [
        fn.replace("-summary","-timeseries")
        for fn in selected_df["filename"]
    ]
    trace_files = list(set(cpu_files + gpu_files))

    # filter by partition
    if cpu_only:
        trace_files = cpu_files
    elif gpu_only:
        trace_files = gpu_files
    # else leave both
    trace_files = list(set(trace_files))

    print(f"Total files to load: {len(trace_files)} (CPU: {len(cpu_files)}, GPU: {len(gpu_files)})")

    # 7) Read SLURM log
    slurm_log = next(
        (
            os.path.join(r, "slurm-log.csv")
            for r, _, fs in os.walk(os.path.join(local_dataset_path, data_subdir))
            if "slurm-log.csv" in fs
        ),
        None
    )
    if not slurm_log:
        return [], 0, 0
    slurm_df = pd.read_csv(slurm_log)

    # 8) Process each file, populating data_dict
    data_dict = {}
    for rel_path in tqdm(trace_files, desc="Processing trace files"):
        fpath = os.path.join(local_dataset_path, data_subdir, rel_path)
        if not os.path.exists(fpath):
            print(f"Warning: missing {fpath}")
            continue

        tqdm.write(f"Reading {rel_path}")
        dfi = pd.read_csv(fpath, dtype={0: str})
        jobid = int(os.path.basename(rel_path).split("-")[0])
        data_dict.setdefault(jobid, {})

        # CPU timeseries
        if rel_path.endswith("-timeseries.csv") and "cpu" not in data_dict[jobid]:
            data_dict[jobid]["cpu"] = proc_cpu_series(dfi)

        # GPU timeseries
        elif "gpu_index" in dfi.columns:
            cpu_df = data_dict[jobid].get("cpu")
            if cpu_df is None:
                continue
            gpu_cnt = data_dict[jobid].get("gpu_cnt", 0)
            prev_gpu = data_dict[jobid].get("gpu")
            gpu_ser, gpu_cnt = proc_gpu_series(cpu_df, dfi, gpu_cnt)
            if prev_gpu is None:
                data_dict[jobid]["gpu"] = gpu_ser
            else:
                data_dict[jobid]["gpu"] = pd.merge(prev_gpu, gpu_ser, on="utime")
            data_dict[jobid]["gpu_cnt"] = gpu_cnt

    # 9) Merge SLURM metadata for each job
    for jobid in list(data_dict):
        matches = slurm_df[slurm_df["id_job"] == jobid]
        if len(matches) == 1:
            data_dict[jobid].update(matches.iloc[0].to_dict())

    # 10) Compute overall time bounds
    cpu_utimes = [d["cpu"]["utime"] for d in data_dict.values() if "cpu" in d]

    # 11) Build the final list of job_dicts
    jobs_list = []
    for jobid, data in data_dict.items():
        # skip any job that never loaded a CPU trace
        cpu_ser = data.get("cpu")
        if cpu_ser is None:
            print(f"Warning: skipping job {jobid} (no CPU trace)")
            continue
        cpu_trace = cpu_ser["cpu_utilisation"]
        cpu_trace = cpu_trace.tolist() if isinstance(cpu_trace, pd.Series) else cpu_trace
        gpu_df = data.get("gpu")
        gpu_trace_list = gpu_df.values.tolist() if isinstance(gpu_df, pd.DataFrame) else 0

        submit_time = data.get("time_submit") - start_ts
        job_start = data["cpu"]["utime"].min() - start_ts
        job_end   = data["cpu"]["utime"].max() - start_ts
        wall_time = max(0, job_end - job_start)
        nodes_req = data.get("nodes_alloc")
        if nodes_req > 1 and cpu_trace:
            cpu_trace = [x / nodes_req for x in cpu_trace]

        jobs_list.append(job_dict(
            nodes_required=nodes_req,
            name=data.get("name_job", "unknown"),
            account=data.get("id_user", "unknown"),
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace_list,
            ntx_trace=[],
            nrx_trace=[],
            end_state=data.get("state_end", "UNKNOWN"),
            id=jobid,
            priority=data.get("priority", 0),
            submit_time=submit_time,
            time_limit=data.get("time_limit", 0),
            start_time=job_start,
            end_time=job_end,
            wall_time=wall_time,
            trace_time=len(cpu_trace) * 10.0,
            trace_start_time=0,
            trace_end_time=len(cpu_trace) * 10.0
        ))

    return jobs_list, 0, requested_duration
