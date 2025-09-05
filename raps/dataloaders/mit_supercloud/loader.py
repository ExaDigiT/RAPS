#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT Supercloud data loader

This module extracts and processes job traces from the MIT SuperCloud dataset,
starting with slurm-log.csv file, and then searching for the files in the cpu
and gpu directories. The main paper associated with the MIT Supercloud Dataset
is available here: https://arxiv.org/abs/2108.02037.
There is more information available here: https://dcc.mit.edu/

Note, that quite a bit of filtering is done with sanity checks to make sure
the the CPU traces match the GPU traces, etc. At this point it's not uncommon
if there may be 1569 total jobs in the time range, only 834 cpu jobs and 128
gpu jobs (962 total) are able to be replayed. This is an issue which will likely
have to be improved in the future.

---------------------------------------------------------------------------
Understanding some of the errors. We track the different reasons that
less than the total number of jobs in the slurm log actually run in the
simulator. This is not so much an issue for the CPU partition, but for
the GPU partition, where we have to combine traces extracted from both
CPU trace files and GPU trace files.

At the beginning of the GPU partition analysis, we give an analysis such as:

    --- Detailed Job Accounting for Partition 'part-gpu' ---
    Initial jobs considered: 519
       * Jobs with NO trace file found: 69 (519 - 450)

    Of the 450 jobs with traces:
       * 289 jobs have only CPU traces (417 - 128)
       * 33 jobs have only GPU traces (161 - 128)
       * 128 jobs have BOTH CPU and GPU traces.
    ----------------------------------------------------

We give a summary report at the end of the data loading process. An
example report is shown for the range `--start 2021-05-21T00:00 --end 2021-05-22T00:00`

    Skipped jobs summary:
    - nodes_alloc > 480: 0
    - pruned_nodes: 1
    - no_trace_file: 69
    - no_cpu_trace_for_gpu_job: 41
    - final_gpu_none_mixed: 289
    - final_cpu_none_mixed: 33

    [INFO] Partition 'mit_supercloud/part-cpu': 834 jobs loaded
    [INFO] Partition 'mit_supercloud/part-gpu': 128 jobs loaded

We explain each of these stats here.

    - `nodes_alloc > 480`: the number of jobs that are thrown out because
       they request more than 480 nodes.

    - `pruned_nodes`: the number of jobs thrown out because the node was
       listed in `prune_list.txt`.

    - `no_trace_file`: the number of jobs that were found in the Slurm log
       for the correct time window and partition, but for which not a single
       corresponding trace file (neither CPU nor GPU) could be found on the filesystem.

    - `no_cpu_trace_for_gpu_job`: The number of jobs that had a GPU trace file
       but were discarded because they were missing their required corresponding
       CPU trace file.

    - `final_gpu_none_mixed`: The number of jobs in a GPU partition run that had
      a CPU trace but were missing the final, processed GPU trace data.

    - `final_cpu_none_mixed`: The number of jobs in a GPU partition run that were
      missing the essential CPU trace data during the final job construction phase.

Now, we work on debugging some of these. For example, for `no_cpu_trace_for_gpu_job`,
we can take the jid from the warning message:

    [WARNING] → no cpu trace for gpu! (jid=4074251073298) SKIPPING

And then check the data directory to see if it can find trace files for both the cpu
and gpu:

    > find ~/data/mit/202201 -name '4074251073298*'

---------------------------------------------------------------------------
How we curated and generated the node ids: cpu_nodes.txt and gpu_nodes.txt

Node filtering based on observed resource allocation history.

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
import pandas as pd
import re

from tqdm import tqdm
from typing import Dict, Union, Optional
from collections import Counter
from datetime import datetime, timezone

from raps.job import job_dict, Job
from raps.utils import summarize_ranges, next_arrival, WorkloadData
from .utils import proc_cpu_series, proc_gpu_series, to_epoch
from .utils import DEFAULT_START, DEFAULT_END

TRES_ID_MAP = {
    1: "cpu",
    2: "mem",     # in MB
    3: "energy",
    4: "gres/gpu",
    5: "billing",
}
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def parse_tres_alloc(tres_str: Union[str, None],
                     id_map: Optional[Dict[int, str]] = None,
                     return_ids: bool = False,
                     stats: Counter = None) -> Dict[Union[int, str], int]:
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
    stats : Counter
        Optional counter to track parsing errors.

    Returns
    -------
    dict
        Parsed key/value pairs. Example:
        {'cpu': 20, 'mem': 170000, 'gres/gpu': 1, 'billing': 20}
    """
    if pd.isna(tres_str):
        return {}
    tres_str = str(tres_str)

    id_map = id_map or TRES_ID_MAP

    # strip quotes or whitespace
    tres_str = tres_str.strip().strip('"').strip("'")

    if not tres_str:
        return {}

    # Split on commas, but be tolerant of spaces
    parts = [p for p in tres_str.split(",") if p]

    out: Dict[Union[int, str], int] = {}

    for p in parts:
        m = re.match(r"\s*(\d+)\s*=\s*([0-9]+)\s*$", p)
        if not m:
            if stats is not None:
                stats["malformed_tres"] += 1
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
    config = kwargs.get("config")
    arrival = kwargs.get("arrival")
    NL_PATH = os.path.dirname(__file__)

    skip_counts = Counter()

    # unpack
    if isinstance(local_dataset_path, list):
        if len(local_dataset_path) != 1:
            raise ValueError("Expect exactly one path")
        local_dataset_path = local_dataset_path[0]

    # slurm log -> DataFrame
    slurm_path = None
    for root, _, files in os.walk(os.path.expanduser(local_dataset_path)):
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
    end_ts = to_epoch(kwargs.get("end",   DEFAULT_END))

    mask = (sl.time_submit >= start_ts) & (sl.time_submit < end_ts)
    sl = sl[mask]

    if debug:
        print(f"[DEBUG] After time filtering: {len(sl)} jobs")
        hits = sl.loc[mask]
        lines = hits["__line__"].tolist()
        print(f"data sourced from {len(lines)} records in slurm-log.csv. Line number ranges:",
              summarize_ranges(lines))

    # --- prune out oversized jobs and known under‑used hosts ---
    # load list of underutilized nodes to ignore
    pruned = set()
    with open(os.path.join(NL_PATH, "prune_list.txt")) as pf:
        pruned = {l_.strip() for l_ in pf if l_.strip()}

    before_prune = len(sl)
    # only keep jobs requesting <= 480 nodes
    sl = sl[sl.nodes_alloc <= 480]
    after_alloc_filter = len(sl)
    skip_counts['nodes_alloc > 480'] += (before_prune - after_alloc_filter)

    # drop any job whose nodelist includes a pruned node
    sl["nodes_list"] = sl["nodelist"].apply(ast.literal_eval)

    def is_pruned(lst):
        matches = [n for n in lst if n in pruned]
        if matches:
            if debug:
                print(f"[DEBUG] Skipping job due to pruned nodes: {matches}")
            return True
        return False

    before_prune_filter = len(sl)
    sl = sl[~sl["nodes_list"].apply(is_pruned)]
    after_prune_filter = len(sl)
    skip_counts['pruned_nodes'] += (before_prune_filter - after_prune_filter)

    if debug:
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
    part = kwargs.get("partition", "").split("/")[-1].lower()
    cpu_only = (part == "part-cpu")
    mixed = (part == "part-gpu")

    # handle single-partition configs (e.g., mit_supercloud.yaml)
    if not cpu_only and not mixed:
        gpus_per_node = config.get("GPUS_PER_NODE")

        if gpus_per_node == 0:
            cpu_only = True
            part = "part-cpu"
        else:
            mixed = True
            part = "part-gpu"

    # create nodelist mapping
    if cpu_only:
        with open(os.path.join(NL_PATH, "cpu_nodes.txt")) as f:
            cpu_nodes = [l_.strip() for l_ in f if l_.strip()]
        cpu_node_to_idx = {h: i for i, h in enumerate(cpu_nodes)}
    else:  # cpu + gpu
        with open(os.path.join(NL_PATH, "gpu_nodes.txt")) as f:
            gpu_nodes = [l_.strip() for l_ in f if l_.strip()]
        gpu_node_to_idx = {h: i for i, h in enumerate(gpu_nodes)}

    if cpu_only:
        job_ids = set(sl.id_job) - gpu_jobs
        # skip_counts['gpu_job_in_cpu_mode'] += len(set(sl.id_job) & gpu_jobs)
    elif mixed:
        job_ids = gpu_jobs & set(sl.id_job)
        # skip_counts['cpu_job_in_gpu_mode'] += len(set(sl.id_job) - gpu_jobs)
    else:
        job_ids = set(sl.id_job)

    print(f"{GREEN}→ mode={part}, jobs: {len(job_ids)}{RESET}")

    # find trace files by walking directories
    cpu_files = []
    cpu_root = os.path.join(data_root, "cpu")
    if os.path.exists(cpu_root):
        for R, _, fs in os.walk(cpu_root):
            for f in fs:
                if not f.endswith("-timeseries.csv"):
                    continue
                try:
                    jid = int(f.split("-", 1)[0])
                    if jid in job_ids:
                        cpu_files.append(os.path.join(R, f))
                except (ValueError, IndexError):
                    continue

    gpu_files = []
    gpu_root = os.path.join(data_root, "gpu")
    if os.path.exists(gpu_root):
        for R, _, fs in os.walk(gpu_root):
            for f in fs:
                if not f.endswith(".csv"):
                    continue
                try:
                    jid = int(f.split("-", 1)[0])
                    if jid in job_ids:
                        gpu_files.append(os.path.join(R, f))
                except (ValueError, IndexError):
                    continue

    cpu_ids = {int(os.path.basename(p).split('-', 1)[0]) for p in cpu_files}
    gpu_ids = {int(os.path.basename(p).split('-', 1)[0]) for p in gpu_files}
    all_trace_ids = cpu_ids | gpu_ids

    print(f"→ {len(cpu_files)} CPU files, {len(gpu_files)} GPU files → {len(all_trace_ids)} jobs with traces")

    if mixed:
        # Perform a full accounting of all jobs considered for the partition.
        jobs_with_no_traces = len(job_ids - all_trace_ids)
        jobs_with_traces = len(all_trace_ids)

        print(f"\n--- Detailed Job Accounting for Partition '{part}' ---")
        print(f"Initial jobs considered: {len(job_ids)}")
        print(f"   * Jobs with NO trace file found: {jobs_with_no_traces} ({len(job_ids)} - {jobs_with_traces})\n")

        if jobs_with_traces > 0:
            overlap_count = len(cpu_ids & gpu_ids)
            cpu_only_count = len(cpu_ids) - overlap_count
            gpu_only_count = len(gpu_ids) - overlap_count
            print(f"Of the {jobs_with_traces} jobs with traces:")
            print(f"   * {cpu_only_count} jobs have only CPU traces ({len(cpu_ids)} - {overlap_count})")
            print(f"   * {gpu_only_count} jobs have only GPU traces ({len(gpu_ids)} - {overlap_count})")
            print(f"   * {overlap_count} jobs have BOTH CPU and GPU traces.")
        print("----------------------------------------------------\n")

    data = {}

    traced_jobs = all_trace_ids
    untraced_jobs = job_ids - traced_jobs
    skip_counts['no_trace_file'] += len(untraced_jobs)

    # CPU first
    for fp in tqdm(cpu_files, desc="Loading CPU traces"):
        df = pd.read_csv(fp, dtype={0: str})
        jid = int(os.path.basename(fp).split("-", 1)[0])
        rec = data.setdefault(jid, {})

        # Find job info in slurm log and print details
        job_info = sl[sl.id_job == jid]
        if job_info.empty:
            skip_counts['job_not_in_slurm_log'] += 1
            if debug:
                tqdm.write(f"Reading CPU {os.path.basename(fp)} for Job ID: {jid} (No slurm info found)")
            continue

        job_row = job_info.iloc[0]
        if debug:
            start_time = job_row.get('time_start', 'N/A')
            wall_time = job_row.get('time_limit', 'N/A')
            tres_alloc = job_row.get('tres_alloc', 'N/A')
            tres_alloc_dict = parse_tres_alloc(tres_alloc)
            rec["tres_alloc_dict"] = tres_alloc_dict
            #  gres_used = job_row.get('gres_used', 'N/A')  # Unused

            tqdm.write(f"Reading CPU {os.path.basename(fp)} for Job ID: {jid}")
            tqdm.write(f"  Start Time: {start_time}, Wall Time: {wall_time}s")
            tqdm.write(f"  TRES Alloc: {tres_alloc_dict}")

        tres_alloc = job_row.get('tres_alloc', 'N/A')
        tres_alloc_dict = parse_tres_alloc(tres_alloc, stats=skip_counts)
        rec["tres_alloc_dict"] = tres_alloc_dict

        raw = job_row.get("nodelist", "")
        hosts = ast.literal_eval(raw)
        # Get allocated nodes "['r9189566-n911952','r9189567-n...']"
        try:
            if cpu_only:
                rec["scheduled_nodes"] = [cpu_node_to_idx[h] for h in hosts]
            else:
                rec["scheduled_nodes"] = [gpu_node_to_idx[h] for h in hosts]
        except KeyError as e:
            skip_counts['unrecognized_node_name'] += 1
            if debug:
                print(f"Skipping job {jid} due to unrecognized node name: {e}")
            continue

        rec["nodes_alloc"] = int(job_row["nodes_alloc"])
        rec["cpu"] = proc_cpu_series(df)
        # print(f'{RED}{rec["cpu"]}{RESET}')

    if debug:
        print(f"GPU candidate files ({len(gpu_files)}):")
        for p in gpu_files[:10]:
            print("   ", p)

    # data from the cpu processes are all stored under the `data` dictionary
    # according to their respective jid key
    # print("******", data.keys())

    for fp in tqdm(gpu_files, desc="Loading GPU traces"):

        if not os.path.exists(fp):
            if debug:
                print(f"{YELLOW}[WARNING] gpu path {fp!r} doesn't exist skipping{RESET}")
            skip_counts['gpu_path_does_not_exist'] += 1
            continue

        if debug:
            tqdm.write(f"Reading GPU {os.path.basename(fp)}")
        dfi = pd.read_csv(fp, dtype={0: str})
        if "gpu_index" not in dfi.columns:
            if debug:
                tqdm.write("[WARNING] → no gpu_index column!  SKIPPING")
            skip_counts['no_gpu_index_column'] += 1
            continue

        jid = int(os.path.basename(fp).split("-", 1)[0])
        rec = data.setdefault(jid, {})
        cpu_df = rec.get("cpu")
        # print(f"{YELLOW}jid={jid} {cpu_df}{RESET}")
        if cpu_df is None:
            if debug:
                tqdm.write(f"{YELLOW}[WARNING] → no cpu trace for gpu! (jid={jid}) SKIPPING{RESET}")
            skip_counts['no_cpu_trace_for_gpu_job'] += 1
            continue

        gpu_cnt = rec.get("gpu_cnt", 0)
        gpu_ser, gpu_cnt = proc_gpu_series(cpu_df, dfi, gpu_cnt)

        gpu_cnt = data[jid].get("gpu_cnt", 0)
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

        # grab all the gpu-util columns
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

    # merge slurm metadata
    for _, row in sl.iterrows():
        jid = row.id_job
        if jid in data and 'id_job' not in data[jid]:
            data[jid].update(row.to_dict())

    # build final job_dicts
    jobs_list = []

    # Get CPUS_PER_NODE and GPUS_PER_NODE from config
    cpus_per_node = config.get('CPUS_PER_NODE')
    cores_per_cpu = config.get('CORES_PER_CPU')
    # gpus_per_node = config.get('GPUS_PER_NODE')  # Unused

    quanta = config.get('TRACE_QUANTA')

    for jid, rec in data.items():
        nr = rec.get("nodes_alloc")
        if nr is None:
            skip_counts['final_missing_nodes_alloc'] += 1
            continue

        cpu = rec.get("cpu")
        gpu = rec.get("gpu_trace")

        cpu_tr = []
        gpu_tr = []
        t0, t1 = 0, 0

        if cpu_only:
            if cpu is None:
                skip_counts['final_cpu_none_cpu_only'] += 1
                continue
            cpu_tr = cpu.cpu_utilisation.tolist()
            gpu_tr = [0]  # Ensure gpu_tr is a list for max() operation
            t0, t1 = cpu.utime.min(), cpu.utime.max()
        elif mixed:
            if cpu is None:
                skip_counts['final_cpu_none_mixed'] += 1
                continue
            if gpu is None:
                skip_counts['final_gpu_none_mixed'] += 1
                continue
            cpu_tr = cpu.cpu_utilisation.tolist()
            gpu_tr = gpu
            t0, t1 = cpu.utime.min(), cpu.utime.max()
        else:  # not cpu_only or mixed
            skip_counts['final_unhandled_partition'] += 1
            continue

        # Calculate cpu_cores_required and gpu_units_required from tres_alloc
        if "tres_alloc_dict" not in rec:
            skip_counts['final_missing_tres_alloc'] += 1
            continue

        total_cpu = rec["tres_alloc_dict"].get('cpu', 0)
        # Can either allocate gpu:volta (1002) or gpu:tesla (1001) but not both
        total_gpu = rec["tres_alloc_dict"].get('1002') or rec["tres_alloc_dict"].get(1001, 0)

        cpu_cores_req = math.ceil(total_cpu / nr)
        gpu_units_req = math.ceil(total_gpu / nr)

        # sometimes there are spurious large values for cpu util - set max limit based on peak
        cpu_peak = cpu_cores_req / cores_per_cpu / cpus_per_node  # Is this per CPU?
        cpu_tr = [min(x/cores_per_cpu/cpus_per_node, cpu_peak) for x in cpu_tr]

        start_time = t0 - start_ts
        end_time = t1 - start_ts
        submit_time = rec.get("time_submit") - start_ts
        scheduled_nodes = rec.get("scheduled_nodes")

        current_job_dict = job_dict(
            nodes_required=nr,
            cpu_cores_required=cpu_cores_req,
            gpu_units_required=gpu_units_req,
            name=rec.get("name_job", "unknown"),
            account=rec.get("id_user", "unknown"),
            cpu_trace=cpu_tr,
            gpu_trace=gpu_tr,
            ntx_trace=[],
            nrx_trace=[],
            end_state=rec.get("state_end", "unknown"),
            id=jid,
            scheduled_nodes=scheduled_nodes,
            priority=rec.get("priority", 0),
            submit_time=submit_time,
            time_limit=rec.get("timelimit") * 60,
            start_time=start_time,
            end_time=end_time,
            expected_run_time=max(0, t1-t0),
            trace_time=len(cpu_tr)*quanta,
            trace_start_time=0,
            trace_end_time=len(cpu_tr)*quanta,
            trace_quanta=quanta
        )
        job = Job(current_job_dict)
        jobs_list.append(job)

    # Calculate min_overall_utime and max_overall_utime
    # min_overall_utime = int(sl.time_submit.min())
    # max_overall_utime = int(sl.time_submit.max())

    # args_namespace = SimpleNamespace(
    #    fastforward=min_overall_utime,
    #    system='mit_supercloud',
    #    time=max_overall_utime
    # )

    print("\nSkipped jobs summary:")
    for reason, count in skip_counts.items():
        print(f"- {reason}: {count}")

    return WorkloadData(
        jobs=jobs_list,
        telemetry_start=0, telemetry_end=int(end_ts - start_ts),
        start_date=datetime.fromtimestamp(start_ts, timezone.utc),
    )
