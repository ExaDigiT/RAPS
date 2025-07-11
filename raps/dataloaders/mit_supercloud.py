#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT Supercloud job trace processing module with load_data function.
"""

import os
import shutil
import sys
from datetime import datetime
import math
from types import SimpleNamespace

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
        tqdm.write(f"Warning: GPU‐CPU time mismatch {per_diff:.1f}% exceeds 10%; continuing anyway")

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
    Load MIT Supercloud job traces **without** any metadata files.
    Expects under:
       local_dataset_path/
         202201/
           cpu/...-timeseries.csv
           gpu/...-timeseries.csv
           slurm-log.csv
    Returns:
       jobs_list, sim_start_time, sim_end_time
    """
    debug = kwargs.get("debug")
    # unpack
    if isinstance(local_dataset_path, list):
        if len(local_dataset_path) != 1:
            raise ValueError("Expect exactly one path")
        local_dataset_path = local_dataset_path[0]

    # 1) slurm log → DataFrame
    sub = "202201"
    slurm_path = os.path.join(local_dataset_path, sub, "slurm-log.csv")
    sl = pd.read_csv(slurm_path)

    # 2) date window
    start_ts = int(datetime.strptime(kwargs.get("start_date","21052021"), "%d%m%Y").timestamp())
    end_ts   = int(datetime.strptime(kwargs.get("end_date",  "22052021"), "%d%m%Y").timestamp())
    duration = end_ts - start_ts

    sl = sl[(sl.time_submit >= start_ts) & (sl.time_submit < end_ts)]

    # 3) detect GPU‐using jobs
    gres = sl.gres_used.fillna("").astype(str)
    tres = sl.tres_alloc.fillna("").astype(str)

    gpu_jobs = set(sl.loc[
        gres.str.contains("gpu", case=False) |
        tres.str.contains(r"(?:1001|1002)=", regex=True),
        "id_job"
    ])

    # 4) partition mode
    part = kwargs.get("partition","").split("/")[-1].lower()
    cpu_only = (part=="part-cpu")
    mixed    = (part=="part-gpu")

    if cpu_only:
        job_ids = set(sl.id_job) - gpu_jobs
    elif mixed:
        job_ids = gpu_jobs & set(sl.id_job)
    else:
        job_ids = set(sl.id_job)

    print(f"→ mode={part}, jobs: {len(job_ids)}")

    # 5) find trace files by walking directories
    cpu_files = []
    cpu_root = os.path.join(local_dataset_path, sub, "cpu")
    for R,_,fs in os.walk(cpu_root):
        for f in fs:
            if not f.endswith("-timeseries.csv"):
                continue
            jid = int(f.split("-",1)[0])
            if jid in job_ids:
                rel = os.path.relpath(os.path.join(R,f), os.path.join(local_dataset_path,sub))
                cpu_files.append(rel)

    gpu_files = []
    gpu_root = os.path.join(local_dataset_path, sub, "gpu")
    for R,_,fs in os.walk(gpu_root):
        for f in fs:
            #if not f.endswith("-timeseries.csv"):
            if not f.endswith(".csv"):
                continue
            jid = int(f.split("-",1)[0])
            if jid in job_ids:
                rel = os.path.relpath(os.path.join(R,f), os.path.join(local_dataset_path,sub))
                gpu_files.append(rel)

    # 6) select final trace list
    if cpu_only:
        traces = cpu_files
    elif mixed:
        traces = list(set(cpu_files + gpu_files))

        ### check overlap
        cpu_ids = {int(os.path.basename(p).split('-',1)[0]) for p in cpu_files}
        gpu_ids = {int(os.path.basename(p).split('-',1)[0]) for p in gpu_files}

        if debug:
            print(f"[DEBUG] CPU IDs: {len(cpu_ids)}  GPU IDs: {len(gpu_ids)}  OVERLAP: {len(cpu_ids & gpu_ids)}")
            if cpu_ids & gpu_ids:
                print("   example overlap:", list(cpu_ids & gpu_ids)[:5])
            else:
                print("   → **No overlap**!  That means none of your GPU job IDs ever had a CPU file in `cpu_files`.")

    else:
        traces = list(set(cpu_files + gpu_files))

    print(f"→ {len(cpu_files)} CPU files, {len(gpu_files)} GPU files → total {len(traces)}")

    data = {}

    # 8a) CPU first
    for rel in tqdm(cpu_files, desc="Loading CPU traces"):
        fp = os.path.join(local_dataset_path, sub, rel)
        df = pd.read_csv(fp, dtype={0: str})
        jid = int(os.path.basename(rel).split("-", 1)[0])
        rec = data.setdefault(jid, {})
        tqdm.write(f"Reading CPU {rel}")
        rec["cpu"] = proc_cpu_series(df)

    print(f"GPU candidate files ({len(gpu_files)}):")
    for p in gpu_files[:10]:
        print("   ", p)

    for rel in tqdm(gpu_files, desc="Loading GPU traces"):
        fp = os.path.join(local_dataset_path, sub, rel)
        if debug:
            print(f"\n[DEBUG] attempting {rel!r}")
            print("        full path exists:", os.path.exists(fp), fp)
        if not os.path.exists(fp):
            continue

        tqdm.write(f"Reading GPU {rel}")
        dfi = pd.read_csv(fp, dtype={0: str})
        if debug:
            print("        loaded dataframe, columns:", dfi.columns.tolist())
        if "gpu_index" not in dfi.columns:
            tqdm.write("        → no gpu_index column!  SKIPPING")
            continue

        jid = int(os.path.basename(rel).split("-", 1)[0])
        rec = data.setdefault(jid, {})
        cpu_df = rec.get("cpu")
        if cpu_df is None:
            tqdm.write(f"Warning: no CPU trace for job {jid}, skipping GPU")
            continue

        gpu_cnt = rec.get("gpu_cnt", 0)
        gpu_ser, gpu_cnt = proc_gpu_series(cpu_df, dfi, gpu_cnt)

        gpu_cnt  = data[jid].get("gpu_cnt", 0)
        prev_gpu = data[jid].get("gpu")   # ← define prev_gpu here
        gpu_ser, gpu_cnt = proc_gpu_series(cpu_df, dfi, gpu_cnt)
        if prev_gpu is None:
            data[jid]["gpu"] = gpu_ser
        else:
            data[jid]["gpu"] = pd.merge(prev_gpu, gpu_ser, on="utime")
        data[jid]["gpu_cnt"] = gpu_cnt

        if debug:
            print(f"[DEBUG] proc_gpu_series returned {len(gpu_ser)} rows (gpu_cnt={gpu_cnt})")

        if "gpu" in rec:
            rec["gpu"] = pd.merge(rec["gpu"], gpu_ser, on="utime", how="outer")
        else:
            rec["gpu"] = gpu_ser
        rec["gpu_cnt"] = gpu_cnt

        gpu_df = rec["gpu"]

        # 1) grab all the gpu‐util columns
        util_cols = [c for c in gpu_df.columns if c.startswith("gpu_util_")]

        if not util_cols:
            # no gpu utilization columns? zero out
            rec["gpu_trace"] = []
        else:
            # 2) as floats in [0,1]
            raw = gpu_df[util_cols].astype(float).div(100)

            # 3) average (or sum) across devices
            #    if you want to SUM instead, use .sum(axis=1)
            avg_util = raw.mean(axis=1)

            # 4) scale by number of nodes requested
            nodes = rec.get("nodes_alloc", 1)
            rec["gpu_trace"] = (avg_util * nodes).tolist()

        if debug:
            print(f"[DEBUG] data[{jid}].keys() now:", list(rec.keys()))

    # quick check: did any jobs pick up a GPU trace?
    print("→ data_dict contents sample:")
    for jid, rec in list(data.items())[:5]:
        print(f"   job {jid}: cpu={'yes' if 'cpu' in rec else 'no'}  gpu={'yes' if 'gpu' in rec else 'no'}")
    print(f"→ total jobs seen = {len(data)}")

    got = [jid for jid, rec in data.items() if "gpu" in rec]
    miss = [jid for jid, rec in data.items() if "cpu" in rec and "gpu" not in rec]
    print(f"→ of {len(data)} total jobs seen, {len(got)} got GPU data, {len(miss)} have only CPU")
    if miss:
        print("   jobs missing GPU despite being in gpu_files:", miss[:10])

    # 8) merge slurm metadata
    for _, row in sl.iterrows():
        jid = row.id_job
        if jid in data and jid not in data[jid]:
            data[jid].update(row.to_dict())

    # 9) build final job_dicts
    jobs_list = []
    
    # Get CPUS_PER_NODE and GPUS_PER_NODE from config
    config = kwargs.get('config', {})
    cpus_per_node = config.get('CPUS_PER_NODE', 2) # Default to 2 if not found
    gpus_per_node = config.get('GPUS_PER_NODE', 0) # Default to 0 if not found

    for jid, rec in data.items():
        cpu = rec.get("cpu")
        gpu = rec.get("gpu_trace")

        cpu_tr = []
        gpu_tr = []
        t0, t1 = 0, 0

        if cpu_only:
            if cpu is None:
                print("cpu None: skipping this one (a)")
                continue
            cpu_tr = cpu.cpu_utilisation.tolist()
            gpu_tr = [0] # Ensure gpu_tr is a list for max() operation
            t0, t1 = cpu.utime.min(), cpu.utime.max()
        elif mixed:
            if cpu is None:
                print("cpu None: skipping this one (b)")
                continue
            if gpu is None:
                print("gpu None: skipping this one")
                continue
            cpu_tr = cpu.cpu_utilisation.tolist()
            gpu_tr = gpu
            t0, t1 = cpu.utime.min(), cpu.utime.max()
        else:
            print("skipping")
            continue

        st = rec.get("time_submit",t0) - start_ts
        nr = rec.get("nodes_alloc",1)
        if nr>1:
            cpu_tr = [x/nr for x in cpu_tr]

        # Calculate cpu_cores_required and gpu_units_required
        cpu_cores_req = math.ceil(max(cpu_tr) * cpus_per_node) if cpu_tr else 0
        gpu_units_req = math.ceil(max(gpu_tr) * gpus_per_node) if gpu_tr else 0

        jobs_list.append(job_dict(
            nodes_required   = nr,
            cpu_cores_required = cpu_cores_req,
            gpu_units_required = gpu_units_req,
            name             = rec.get("name_job","unknown"),
            account          = rec.get("id_user","unknown"),
            cpu_trace        = cpu_tr,
            gpu_trace        = gpu_tr,
            ntx_trace        = [],
            nrx_trace        = [],
            end_state        = rec.get("state_end","UNKNOWN"),
            id               = jid,
            priority         = rec.get("priority",0),
            submit_time      = st,
            time_limit       = rec.get("time_limit",0),
            start_time       = t0 - start_ts,
            end_time         = t1 - start_ts,
            wall_time        = max(0, t1-t0),
            trace_time       = len(cpu_tr)*10.0,
            trace_start_time = 0,
            trace_end_time   = len(cpu_tr)*10.0
        ))

    # Calculate min_overall_utime and max_overall_utime
    min_overall_utime = int(sl.time_submit.min())
    max_overall_utime = int(sl.time_submit.max())

    args_namespace = SimpleNamespace(
        fastforward=min_overall_utime,
        system='mit_supercloud',
        time=max_overall_utime
    )

    return jobs_list, min_overall_utime, max_overall_utime, args_namespace
