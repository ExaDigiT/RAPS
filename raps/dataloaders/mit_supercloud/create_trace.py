#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified and PEP-8 compliant refactor of the original script.

Original script created on Fri Sep 20 10:14:23 2024 by Damien Fay (HPE)
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix as csr
from tqdm import tqdm

# Add the raps project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

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

def main(local_dataset_path, start_date_str, end_date_str):
    mit_dir = os.path.dirname(os.path.abspath(__file__))
    tracedir = os.path.join(mit_dir, 'data', 'trace')
    os.makedirs(tracedir, exist_ok=True)

    start_ts = int(datetime.strptime(start_date_str, '%d%m%Y').timestamp())
    end_ts = int(datetime.strptime(end_date_str, '%d%m%Y').timestamp())

    file_df = pd.read_csv(os.path.join(mit_dir, 'source_data', 'file_list.csv'), sep='\t')
    gpu_df = file_df[file_df['File Name'].str.contains('/gpu/')].copy()
    gpu_df['jobid'] = gpu_df['File Name'].str.extract(r'/([^/]+?)-').astype(int)

    job_df = pd.read_csv(os.path.join(mit_dir, 'source_data', 'job_user_date_full.csv'))
    selected_df = job_df[(job_df.start > start_ts) & (job_df.start < end_ts)].copy()

    files_to_copy = [row['filename'].replace('-summary', '-timeseries') for _, row in selected_df.iterrows()]
    files_to_copy += gpu_df[gpu_df.jobid.isin(selected_df.job_id)]['File Name'].tolist()
    files_to_copy = list(set(files_to_copy))

    for rel_path in tqdm(files_to_copy, desc="Copying trace files"):
        src = os.path.join(local_dataset_path, rel_path)
        dst = os.path.join(tracedir, os.path.basename(rel_path))
        if not os.path.exists(src):
            print(f"Missing: {src}")
            continue
        if os.path.exists(dst) and os.path.getsize(src) == os.path.getsize(dst):
            continue
        shutil.copy2(src, dst)

    slurm_log = None
    for root, _, files in os.walk(local_dataset_path):
        if 'slurm-log.csv' in files:
            slurm_log = os.path.join(root, 'slurm-log.csv')
            break
    if not slurm_log:
        print(f"slurm-log.csv not found in {local_dataset_path}.")
        return

    slurm_df = pd.read_csv(slurm_log)
    traced_files = sorted(f for f in os.listdir(tracedir) if 'lock' not in f)
    print(f"Processing {len(traced_files)} trace files.")

    data_dict = {}
    for idx, s in enumerate(traced_files):
        if idx % 100 == 0:
            print(f"processing file {idx} of {len(traced_files)}")
        fpath = os.path.join(tracedir, s)
        dfi = pd.read_csv(fpath, dtype={0: str})
        jobid = int(s.split('-')[0])

        if jobid not in data_dict:
            data_dict[jobid] = {}
            slurm_idx = np.where(slurm_df['id_job'] == jobid)[0]
            if slurm_idx.shape[0] != 1:
                continue
            data_dict[jobid] = slurm_df.iloc[slurm_idx[0]].to_dict()

        if 'timeseries' in s:
            if 'cpu' in data_dict[jobid]:
                continue
            cpu_ser = proc_cpu_series(dfi)
            data_dict[jobid]['cpu'] = cpu_ser

        elif 'gpu_index' in dfi.columns:
            rack = s.split('-')[1]
            node = s.split('-')[2].split('.csv')[0]
            cpu_df = data_dict[jobid].get('cpu')
            if cpu_df is None:
                continue

            gpu_cnt = data_dict[jobid].get('gpu_cnt', 0)
            gpu_df = data_dict[jobid].get('gpu')
            gpu_ser, gpu_cnt = proc_gpu_series(cpu_df, dfi, gpu_cnt)

            if gpu_df is None:
                data_dict[jobid]['gpu'] = gpu_ser
                data_dict[jobid]['grack'] = [rack]
                data_dict[jobid]['gnode'] = [node]
            else:
                data_dict[jobid]['gpu'] = pd.merge(gpu_df, gpu_ser, on='utime')
                data_dict[jobid]['grack'].append(rack)
                data_dict[jobid]['gnode'].append(node)

            data_dict[jobid]['gpu_cnt'] = gpu_cnt

    print("determining start time...")
    min_utime = min(data['cpu']['utime'].min() for data in data_dict.values() if 'cpu' in data)
    max_utime = max(data['cpu']['utime'].max() for data in data_dict.values() if 'cpu' in data)
    total_sim_time = max_utime - min_utime

    jobs_list = []
    for jobid, data in data_dict.items():
        cpu_trace = data.get('cpu', {}).get('cpu_utilisation', [])
        if isinstance(cpu_trace, pd.Series):
            cpu_trace = cpu_trace.tolist()

        gpu_trace = data.get('gpu')
        gpu_trace_list = gpu_trace.values.tolist() if isinstance(gpu_trace, pd.DataFrame) else 0

        job_start_time = data['cpu']['utime'].min() - min_utime
        job_end_time = data['cpu']['utime'].max() - min_utime
        wall_time = max(0, job_end_time - job_start_time)
        nodes_required = max(1, int(np.ceil(max(cpu_trace) / 2.0))) if cpu_trace else 1
        if nodes_required > 1 and cpu_trace:
            cpu_trace = [x / nodes_required for x in cpu_trace]

        job = job_dict(
            nodes_required=nodes_required,
            name=data.get('name_job', 'unknown'),
            account=data.get('name_account', 'unknown'),
            cpu_trace=cpu_trace,
            gpu_trace=gpu_trace_list,
            ntx_trace=[],
            nrx_trace=[],
            end_state=data.get('state_end', 'UNKNOWN'),
            id=jobid,
            submit_time=job_start_time,
            time_limit=data.get('time_limit', 0),
            start_time=job_start_time,
            end_time=job_end_time,
            wall_time=wall_time,
            trace_time=len(cpu_trace) * 10.0,
            trace_start_time=0,
            trace_end_time=len(cpu_trace) * 10.0
        )
        jobs_list.append(job)

    tf1 = datetime.fromtimestamp(start_ts).strftime('%d_%m_%Y')
    tf2 = datetime.fromtimestamp(end_ts).strftime('%d_%m_%Y')
    save_path = os.path.join(mit_dir, 'data', f'mit_supercloud_jobs_{tf1}__{tf2}.npz')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(
        save_path,
        jobs=np.array(jobs_list),
        start_timestep=0,
        end_timestep=total_sim_time,
        args=SimpleNamespace(fastforward=None, system='mit_supercloud', time=total_sim_time)
    )
    print(f"Saved {len(jobs_list)} jobs to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job trace data from MIT Supercloud logs.")
    parser.add_argument("local_dataset_path", type=str, help="Path to the dataset root.")
    parser.add_argument("--start_date", default="21052021", help="Start date in DDMMYYYY format.")
    parser.add_argument("--end_date", default="22052021", help="End date in DDMMYYYY format.")
    args = parser.parse_args()
    main(args.local_dataset_path, args.start_date, args.end_date)
