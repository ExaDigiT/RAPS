#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT Supercloud data loader

This module processes job traces from the MIT SuperCloud dataset with careful
node filtering based on observed resource allocation history.

Summary of node filtering:

- A total of 1135 unique node IDs were extracted from `slurm-log.csv`.
- Of these, 228 were identified as GPU-capable nodes (recorded in `gpu_nodes.txt`).
- The remaining 907 nodes were treated as CPU-only candidates.

Filtering steps:

1. Jobs with `nodes_alloc > 480` were excluded, based on the assumption that 
   such large allocations span across GPU nodes. This removed 413 nodes, 
   leaving 494 candidate CPU-only nodes.

2. To reach the target of 480 CPU nodes, we analyzed job frequency per node 
   and pruned the 14 least-used nodes (those with only 1–26 jobs). 
   These pruned nodes are listed in `prune_list.txt`.

The final list of CPU-only nodes is stored in `cpu_nodes.txt`, and the list
of GPU nodes are stored in `gpu_nodes.txt`.

Note: To locate the pruning logic, search for the keyword "prune" in the code.
"""

import ast
import os
import math
import numpy as np
import pandas as pd
import re

from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, Union, Optional

from raps.job import job_dict, Job
from raps.utils import summarize_ranges
from .utils import proc_cpu_series, proc_gpu_series, to_epoch
from .utils import DEFAULT_START, DEFAULT_END
from .utils import validate_job_traces

TRES_ID_MAP = {
    1: "cpu",
    2: "mem",     # in MB
    3: "energy",
    4: "gres/gpu",
    5: "billing",
}


def parse_tres_alloc(tres_str: Union[str, None],
                     id_map: Optional[Dict[int, str]] = None,
                     return_ids: bool = False) -> Dict[Union[int, str], int]:
    """
    Parse a Slurm tres_alloc/tres_req field like: '1=20,2=170000,4=1,5=20'
    
    Parameters
    ----------
    tres_str : str | None
        The raw TRES string from Slurm (quotes OK). If None/empty returns {}.
    id_map : dict[int,str] | None
        Optional mapping from TRES numeric IDs to friendly names.
        Falls back to TRES_ID_MAP if not provided.
    return_ids : bool
        If True, keys are the numeric IDs. If False, keys use id_map names
        (falls back to the numeric ID as a string if unknown).

    Returns
    -------
    dict
        Parsed key/value pairs. Example:
        {'cpu': 20, 'mem': 170000, 'gres/gpu': 1, 'billing': 20}
    """
    if not tres_str:
        return {}

    id_map = id_map or TRES_ID_MAP

    # strip quotes or whitespace
    tres_str = tres_str.strip().strip('"').strip("'")

    # Split on commas, but be tolerant of spaces
    parts = [p for p in tres_str.split(",") if p]

    out: Dict[Union[int, str], int] = {}

    for p in parts:
        m = re.match(r"\s*(\d+)\s*=\s*([0-9]+)\s*$", p)
        if not m:
            # skip or raise; here we skip silently
            continue
        tid = int(m.group(1))
        val = int(m.group(2))
        if return_ids:
            out[tid] = val
        else:
            key = id_map.get(tid, str(tid))
            out[key] = val

    return out


def load_data(local_dataset_path, **kwargs):
    """
    Load MIT Supercloud job traces **without** any metadata files.
    Expects under:
       local_dataset_path/
         [.../]
           slurm-log.csv
           cpu/...-timeseries.csv
           gpu/...-timeseries.csv
    Returns:
       jobs_list, sim_start_time, sim_end_time
    """
    debug = kwargs.get("debug")
    NL_PATH = os.path.dirname(__file__)
    # unpack
    if isinstance(local_dataset_path, list):
        if len(local_dataset_path) != 1:
            raise ValueError("Expect exactly one path")
        local_dataset_path = local_dataset_path[0]

    # slurm log -> DataFrame
    slurm_path = None
    for root, _, files in os.walk(local_dataset_path):
        if "slurm-log.csv" in files:
            slurm_path = os.path.join(root, "slurm-log.csv")
            break

    if not slurm_path:
        raise FileNotFoundError(f"Could not find slurm-log.csv under {local_dataset_path}")

    data_root = os.path.dirname(slurm_path)
    sl = pd.read_csv(slurm_path)
    sl["__line__"] = sl.index + 2

    # date window
    start_ts = to_epoch(kwargs.get("start", DEFAULT_START))
    end_ts   = to_epoch(kwargs.get("end",   DEFAULT_END))
    #duration = end_ts - start_ts
    
    mask = (sl.time_submit >= start_ts) & (sl.time_submit < end_ts)
    sl = sl[mask]
    print(f"[DEBUG] After time filtering: {len(sl)} jobs")
    hits = sl.loc[mask]
    print("line numbers in slurm-log.csv", summarize_ranges(hits["__line__"].tolist()))

    # --- prune out oversized jobs and known under‑used hosts ---
    # load list of underutilized nodes to ignore
    pruned = set()
    with open(os.path.join(NL_PATH, "prune_list.txt")) as pf:
        pruned = {l.strip() for l in pf if l.strip()}
    print(pruned)
    # only keep jobs requesting ≤480 nodes
    sl = sl[ sl.nodes_alloc <= 480 ]
    print(f"[DEBUG] After nodes_alloc ≤ 480 filter: {len(sl)} jobs")
    # drop any job whose nodelist includes a pruned node
    sl["nodes_list"] = sl["nodelist"].apply(ast.literal_eval)

    def is_pruned(lst):
        matches = [n for n in lst if n in pruned]
        if matches:
            print(f"[DEBUG] Skipping job due to pruned nodes: {matches}")
            return True
        return False

    before = len(sl)
    sl = sl[~sl["nodes_list"].apply(is_pruned)]
    after = len(sl)

    print(f"[DEBUG] Jobs removed by pruning: {before - after}")
    print(f"[DEBUG] After pruning: {len(sl)} jobs")

    # —— ERROR CATCH: no jobs in this window? ——
    if sl.empty:
        raise ValueError(
            f"No SLURM jobs found between {kwargs.get('start_date')} and "
            f"{kwargs.get('end_date')}. Please pick a range covered by the dataset."
        )

    # detect GPU‐using jobs
    gres = sl.gres_used.fillna("").astype(str)
    tres = sl.tres_alloc.fillna("").astype(str)

    gpu_jobs = set(sl.loc[
        gres.str.contains("gpu", case=False) |
        tres.str.contains(r"(?:1001|1002)=", regex=True),
        "id_job"
    ])

    # partition mode
    part = kwargs.get("partition","").split("/")[-1].lower()
    cpu_only = (part=="part-cpu")
    mixed    = (part=="part-gpu")

    # create nodelist mapping
    if cpu_only:
        with open(os.path.join(NL_PATH, "cpu_nodes.txt")) as f:
            cpu_nodes = [l.strip() for l in f if l.strip()]
        cpu_node_to_idx = {h: i for i, h in enumerate(cpu_nodes)}
    else: # cpu + gpu
        with open(os.path.join(NL_PATH, "gpu_nodes.txt")) as f:
            gpu_nodes = [l.strip() for l in f if l.strip()]
        gpu_node_to_idx = {h: i for i, h in enumerate(gpu_nodes)}

    if cpu_only:
        job_ids = set(sl.id_job) - gpu_jobs
    elif mixed:
        job_ids = gpu_jobs & set(sl.id_job)
    else:
        job_ids = set(sl.id_job)

    print(f"→ mode={part}, jobs: {len(job_ids)}")

    # find trace files by walking directories
    cpu_files = []
    cpu_root = os.path.join(data_root, "cpu")
    if os.path.exists(cpu_root):
        for R,_,fs in os.walk(cpu_root):
            for f in fs:
                if not f.endswith("-timeseries.csv"):
                    continue
                try:
                    jid = int(f.split("-",1)[0])
                    if jid in job_ids:
                        cpu_files.append(os.path.join(R,f))
                except (ValueError, IndexError):
                    continue

    gpu_files = []
    gpu_root = os.path.join(data_root, "gpu")
    if os.path.exists(gpu_root):
        for R,_,fs in os.walk(gpu_root):
            for f in fs:
                if not f.endswith(".csv"):
                    continue
                try:
                    jid = int(f.split("-",1)[0])
                    if jid in job_ids:
                        gpu_files.append(os.path.join(R,f))
                except (ValueError, IndexError):
                    continue

    # select final trace list
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

    # CPU first
    for fp in tqdm(cpu_files, desc="Loading CPU traces"):
        df = pd.read_csv(fp, dtype={0: str})
        jid = int(os.path.basename(fp).split("-", 1)[0])
        rec = data.setdefault(jid, {})

        # Find job info in slurm log and print details
        job_info = sl[sl.id_job == jid]
        if not job_info.empty:
            job_row = job_info.iloc[0]
            start_time = job_row.get('time_start', 'N/A')
            wall_time = job_row.get('time_limit', 'N/A')
            tres_alloc = job_row.get('tres_alloc', 'N/A')
            tres_alloc_dict = parse_tres_alloc(tres_alloc)
            rec["tres_alloc_dict"] = tres_alloc_dict
            gres_used = job_row.get('gres_used', 'N/A')

            tqdm.write(f"Reading CPU {os.path.basename(fp)} for Job ID: {jid}")
            tqdm.write(f"  Start Time: {start_time}, Wall Time: {wall_time}s")
            tqdm.write(f"  TRES Alloc: {tres_alloc_dict}")
            #tqdm.write(f"  GRES Used: {gres_used}")
        else:
            tqdm.write(f"Reading CPU {os.path.basename(fp)} for Job ID: {jid} (No slurm info found)")

        raw = job_row.get("nodelist", "")
        hosts = ast.literal_eval(raw)
        # Get allocated nodes "['r9189566-n911952','r9189567-n...']"
        if cpu_only:
            rec["scheduled_nodes"] = [cpu_node_to_idx[h] for h in hosts]
        else:
            rec["scheduled_nodes"] = [gpu_node_to_idx[h] for h in hosts]

        rec["nodes_alloc"] = int(job_row["nodes_alloc"])
        rec["cpu"] = proc_cpu_series(df)

    print(f"GPU candidate files ({len(gpu_files)}):")
    for p in gpu_files[:10]:
        print("   ", p)

    for fp in tqdm(gpu_files, desc="Loading GPU traces"):
        if debug:
            print(f"\n[DEBUG] attempting {fp!r}")
            print("        full path exists:", os.path.exists(fp), fp)
        if not os.path.exists(fp):
            print("gpu path doesn't exist skipping") 
            continue

        tqdm.write(f"Reading GPU {os.path.basename(fp)}")
        dfi = pd.read_csv(fp, dtype={0: str})
        if debug:
            print("        loaded dataframe, columns:", dfi.columns.tolist())
        if "gpu_index" not in dfi.columns:
            tqdm.write("        → no gpu_index column!  SKIPPING")
            continue

        jid = int(os.path.basename(fp).split("-", 1)[0])
        rec = data.setdefault(jid, {})
        cpu_df = rec.get("cpu")
        if cpu_df is None:
            tqdm.write(f"Warning: no CPU trace for job {jid}, skipping GPU")
            continue

        gpu_cnt = rec.get("gpu_cnt", 0)
        gpu_ser, gpu_cnt = proc_gpu_series(cpu_df, dfi, gpu_cnt)

        gpu_cnt  = data[jid].get("gpu_cnt", 0)
        prev_gpu = data[jid].get("gpu")
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

        # grab all the gpu‐util columns
        util_cols = [c for c in gpu_df.columns if c.startswith("gpu_util_")]

        if not util_cols:
            # no gpu utilization columns? zero out
            rec["gpu_trace"] = []
        else:
            # as floats in [0,1]
            raw = gpu_df[util_cols].astype(float).div(100)

            # average across devices
            avg_util = raw.mean(axis=1)

            # scale by number of nodes requested
            nodes = rec.get("nodes_alloc")
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

    # merge slurm metadata
    for _, row in sl.iterrows():
        jid = row.id_job
        if jid in data and jid not in data[jid]:
            data[jid].update(row.to_dict())

    # build final job_dicts
    jobs_list = []
    
    # Get CPUS_PER_NODE and GPUS_PER_NODE from config
    config = kwargs.get('config', {})
    cpus_per_node = config.get('CPUS_PER_NODE')
    cores_per_cpu = config.get('CORES_PER_CPU')
    gpus_per_node = config.get('GPUS_PER_NODE')
    print(f"*** cpus_per_node: {cpus_per_node}, cores_per_cpu: {cores_per_cpu}, gpus_per_node: {gpus_per_node}")

    quanta = config.get('TRACE_QUANTA')

    for jid, rec in data.items():
        nr = rec.get("nodes_alloc")

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

        # Calculate cpu_cores_required and gpu_units_required from tres_alloc
        total_cpu = rec["tres_alloc_dict"].get('cpu', 0)
        # Can either allocate gpu:volta (1002) or gpu:tesla (1001) but not both
        total_gpu = rec["tres_alloc_dict"].get('1002') or tres_alloc_dict.get(1001, 0)

        cpu_cores_req = math.ceil(total_cpu / nr)
        gpu_units_req = math.ceil(total_gpu / nr)

        print(f"*** nr: {nr}, cpu_cores_req: {cpu_cores_req}, gpu_units_req: {gpu_units_req}", flush=True)
        print(jid, cpu_tr[:5], flush=True)

        # sometimes there are spurious large values for cpu util - set max limit based on peak
        cpu_peak = cpu_cores_req / cores_per_cpu / cpus_per_node
        cpu_tr = [min(x/cores_per_cpu/cpus_per_node, cpu_peak) for x in cpu_tr]
        print(jid, cpu_tr[:5])

        submit_time = rec.get("time_submit", t0) - start_ts

        job = job_dict(
            nodes_required   = nr,
            cpu_cores_required = cpu_cores_req,
            gpu_units_required = gpu_units_req,
            name             = rec.get("name_job", "unknown"),
            account          = rec.get("id_user", "unknown"),
            cpu_trace        = cpu_tr,
            gpu_trace        = gpu_tr,
            ntx_trace        = [],
            nrx_trace        = [],
            end_state        = rec.get("state_end", "unknown"),
            id               = jid,
            scheduled_nodes  = rec.get("scheduled_nodes"),
            priority         = rec.get("priority", 0),
            submit_time      = submit_time,
            time_limit       = rec.get("time_limit", 0),
            start_time       = t0 - start_ts,
            end_time         = t1 - start_ts,
            wall_time        = max(0, t1-t0),
            trace_time       = len(cpu_tr)*quanta,
            trace_start_time = 0,
            trace_end_time   = len(cpu_tr)*quanta
        )

        view = job.copy()
        #view['cpu_trace'] = view['cpu_trace'][:5] + ['…']
        #view['gpu_trace'] = view['gpu_trace'][:5] + ['…']

        summarize_trace = lambda x: {
            'min': float(np.min(x)),
            'max': float(np.max(x)),
            'avg': float(np.mean(x)),
            'len': len(x),
        }
        view['cpu_trace'] = summarize_trace(job['cpu_trace'])
        view['gpu_trace'] = summarize_trace(job['gpu_trace'])
        view['cpu_peak'] = job['cpu_cores_required'] / cores_per_cpu / cpus_per_node
        print(view)

        #validate_job_traces(Job(job), granularity=quanta)
        # if nr > 1: # uncomment to test multinode jobs - need to run for 24 hours to get enough jobs to populate
        jobs_list.append(job)

    # Calculate min_overall_utime and max_overall_utime
    min_overall_utime = int(sl.time_submit.min())
    max_overall_utime = int(sl.time_submit.max())

    args_namespace = SimpleNamespace(
        fastforward=min_overall_utime,
        system='mit_supercloud',
        time=max_overall_utime
    )

    return jobs_list, min_overall_utime, max_overall_utime, args_namespace
