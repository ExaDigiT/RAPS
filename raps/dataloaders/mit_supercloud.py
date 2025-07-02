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
    t_cpu = np.array([cpu_df.utime.min(), cpu_df.utime.max()])
    t_gpu = np.array([dfi.timestamp.astype(int).min(), dfi.timestamp.astype(int).max()])

    t_cpu_range = t_cpu[1] - t_cpu[0]
    t_gpu_range = t_gpu[1] - t_gpu[0]
    per_diff = (t_cpu_range - t_gpu_range) / t_gpu_range * 100

    if abs(per_diff) > 10:
        raise ValueError("Time mismatch between CPU and GPU series exceeds 10%")

    dfi['t_fixed'] = dfi.timestamp - dfi.timestamp.min() + t_cpu[0]
    ugpus = dfi.gpu_index.unique()
    gpu_df = pd.DataFrame({'utime': cpu_df['utime'].values})

    for u in ugpus:
        dfg = dfi[dfi.gpu_index == u].copy()
        fields = ['gpu_index', 'utilization_gpu_pct', 'utilization_memory_pct', 'memory_free_MiB',
                  'memory_used_MiB', 'temperature_gpu', 'temperature_memory', 'power_draw_W']

        for field in fields:
            x1, y1 = dfg['t_fixed'].values, dfg[field].values
            xv = cpu_df['utime'].values
            yv = np.interp(xv, x1, y1)
            gpu_df[field] = yv

        rename = {
            'utilization_gpu_pct': f'gpu_{gpu_cnt}',
            'utilization_memory_pct': f'gpu_mem_{gpu_cnt}',
            'temperature_gpu': f'gpu_temp_{gpu_cnt}',
            'power_draw_W': f'gpu_p_{gpu_cnt}'
        }
        gpu_df.rename(columns=rename, inplace=True)
        gpu_cnt += 1

    return gpu_df, gpu_cnt


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
      jobs_list, min_utime, max_utime
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

    start_ts = int(datetime.strptime(start_date_str, "%d%m%Y").timestamp())
    end_ts   = int(datetime.strptime(end_date_str,   "%d%m%Y").timestamp())

    # 4) Select jobs in time window
    selected_df = job_index_df[
        (job_index_df.start > start_ts) &
        (job_index_df.start < end_ts)
    ].copy()

    # 5) Prepare GPU index metadata
    gpu_df = file_list_df[file_list_df["File Name"].str.contains("/gpu/")].copy()
    gpu_df["jobid"] = gpu_df["File Name"].str.extract(r"/([^/]+?)-").astype(int)

    # 6) Build list of timeseries file paths (relative)
    files_to_copy = [
        row["filename"].replace("-summary", "-timeseries")
        for _, row in selected_df.iterrows()
    ]
    files_to_copy += gpu_df[gpu_df.jobid.isin(selected_df.job_id)]["File Name"].tolist()
    files_to_copy = list(set(files_to_copy))

    # 7) Read SLURM log
    data_subdir = "202201"  # hard-coded folder name
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
    for rel_path in tqdm(files_to_copy, desc="Processing trace files"):
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
    #if not cpu_utimes:
    #    return [], 0, 0
    min_utime = min(series.min() for series in cpu_utimes)
    max_utime = max(series.max() for series in cpu_utimes)

    # 11) Build the final list of job_dicts
    jobs_list = []
    for jobid, data in data_dict.items():
        cpu_trace = data["cpu"]["cpu_utilisation"]
        cpu_trace = cpu_trace.tolist() if isinstance(cpu_trace, pd.Series) else cpu_trace
        gpu_df = data.get("gpu")
        gpu_trace_list = gpu_df.values.tolist() if isinstance(gpu_df, pd.DataFrame) else 0

        submit_time = data.get("time_submit") - min_utime
        job_start = data["cpu"]["utime"].min() - min_utime
        job_end   = data["cpu"]["utime"].max() - min_utime
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

    duration = max_utime - min_utime
    return jobs_list, 0, duration
