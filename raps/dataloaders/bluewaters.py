"""
Blue Waters dataloader

Example test case:

    python main.py -f /opt/data/bluewaters --start 20170328 --system bluewaters -net

To download the necessary datasets:

    https://bluewaters.ncsa.illinois.edu/data-sets.html - this explains each of the datasets in detail

    There are two datasets available from:

       https://app.globus.org/file-manager?origin_id=854c1a5c-fa9f-4df4-a71c-407a33e44da0

       1. /torque_logs_anonimized (sic) - we are using the file 2017.tar.gz (377MB)

       2. /node_metrics/cray_system_sampler - we are using the file 20170328.tgz (485MB)

    In order to speed up data loading, we have downsized these files to just
    four columns using the following code:

        import csv
        with open("20170328", "r") as infile, open("output.csv", "w", newline="") as outfile:
            reader = csv.reader(infile, skipinitialspace=True)
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow([row[0], row[1], row[15], row[16]])

    Another dataset we plan to use (but not currently using) is:

       3. Monet - Blue Waters Network Dataset (140GB) - https://databank.illinois.edu/datasets/IDB-2921318

    We assume these datasets are setup as follows (assuming -f /opt/data/bluewaters):

        /opt/data/bluewaters/cray_system_sampler/20170328
        /opt/data/bluewaters/torque_logs/20170328
        /opt/data/bluewaters/monet/20170328
"""

import math
import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from pprint import pprint
from raps.telemetry import Job, job_dict
from raps.utils import WorkloadData


def throughput_traces(total_tx, total_rx, intervals):
    intervals = max(1, int(intervals or 1))
    tx = [(total_tx or 0) // intervals] * intervals
    rx = [(total_rx or 0) // intervals] * intervals
    # print(total_tx, total_rx, intervals, tx[:5], rx[:5])
    return tx, rx


def build_sampler_df(root, day, nodes, tmin, tmax, tx_idx, rx_idx, chunksize=None):
    """One-time loader: returns a DataFrame of per-node positive deltas with mid-interval timestamps.
    Columns: nid, mid_ts, dtx, drx (all numeric)."""
    sdir = Path(root) / "cray_system_sampler" / day
    files = [sdir] if sdir.is_file() else (sorted(f for f in sdir.iterdir() if f.is_file()) if sdir.exists() else [])
    if not files:
        raise FileNotFoundError(f"No Cray sampler files for day {day} under {sdir.parent}")

    cols = [0, 1, tx_idx, rx_idx]  # ts, nid, tx, rx
    out = []

    def _process(df):
        if df.empty:
            return None
        df = df[cols]
        df.columns = ["ts", "nid", "tx", "rx"]
        df = df[df["nid"].isin(nodes)]
        if df.empty:
            return None
        # sort values (optional, for consistency)
        df = df.sort_values(["nid", "ts"])
        # keep raw values
        df = df[["nid", "ts", "tx", "rx"]].dropna()
        return df

    for fp in files:
        print(f"reading {fp}... this may take a while")
        if chunksize:
            for chunk in pd.read_csv(fp, header=None, skipinitialspace=True, chunksize=chunksize):
                dfp = _process(chunk)
                if dfp is not None:
                    out.append(dfp)
        else:
            df = pd.read_csv(fp, header=None, skipinitialspace=True)
            dfp = _process(df)
            if dfp is not None:
                out.append(dfp)

    if not out:
        # nothing matched nodes/time; return empty frame with expected columns
        return pd.DataFrame(columns=["nid", "mid_ts", "dtx", "drx"])

    return pd.concat(out, ignore_index=True)


def hms_to_seconds(wt: str) -> int:
    try:
        h, m, s = map(int, wt.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


def extract_nodes_from_line(hosts_field: str):
    """Extract node IDs from an exec_host field in one line."""
    nodes = []
    for token in hosts_field.split("+"):
        if "/" in token:
            node = token.split("/")[0]
            try:
                nodes.append(int(node))
            except ValueError:
                pass
    return nodes


# example line:
# 03/18/2017 00:01:15;E;6335144.bw;user=USER260243U group=GRP113775G
# account=A116610A jobname=dm5-8506-M9 queue=normal ctime=1489668573
# qtime=1489668573 etime=1489798453 start=1489799118
# owner=USER260243U@h2ologin2
# exec_host=26742/0-31+26743/0-31+26728/0-31+26729/0-31
# login_node=nid27563 Resource_List.flags=commtolerant:commlocal
# Resource_List.neednodes=4:ppn=32 Resource_List.nodect=4
# Resource_List.nodes=4:ppn=32 Resource_List.partition=bwsched
# Resource_List.walltime=04:00:00 session=16472 total_execution_slots=128
# unique_node_count=4 end=1489813275 Exit_status=2 resources_used.cput=28
# resources_used.energy_used=0 resources_used.mem=18996kb
# resources_used.vmem=130088kb resources_used.walltime=03:55:49


PATS = {
    "id": re.compile(r"\b(jobid|job_id|Job_Id)[:=]\s*([^\s,]+)", re.I),
    "name": re.compile(r"\b(jobname)[:=]\s*([^\s,]+)", re.I),
    "account": re.compile(r"\b(account)[:=]\s*([^\s,]+)", re.I),
    # Nodes: use Resource_List.nodect or unique_node_count
    "nodes_required": re.compile(r"\b(?:Resource_List\.nodect|unique_node_count)[:=]\s*(\d+)", re.I),
    # CPU cores per node: from ppn in Resource_List.nodes
    "cpu_cores_required": re.compile(r"\bppn=(\d+)", re.I),
    # GPUs per node
    "gpu_units_required": re.compile(r"\bgpus?=(\d+)", re.I),
    # Scheduled nodes list (exec_host=...)
    "scheduled_nodes": re.compile(r"\bexec_host=([^\s,]+)", re.I),
    # Times
    "submit_time": re.compile(r"\bqtime=([0-9]+)", re.I),
    "start_time": re.compile(r"\bstart=([0-9]+)", re.I),
    "end_time": re.compile(r"\bend=([0-9]+)", re.I),
    # Walltime used
    "wall_time": re.compile(r"resources_used\.walltime=(\d{2}:\d{2}:\d{2})", re.I),
}


def _parse_line(line: str, debug=False):
    rec = {}
    for key, pat in PATS.items():
        m = pat.search(line)
        if m:
            if debug:
                print(f"\n[{key}] matched pattern {pat.pattern}")
                for i in range(0, (m.lastindex or 0) + 1):
                    print(f"  group({i}): {m.group(i)}")
            rec[key] = m.group(m.lastindex or 0)  # take last group
    # normalize scheduled_nodes into list
    if "scheduled_nodes" in rec:
        rec["scheduled_nodes"] = extract_nodes_from_line(rec["scheduled_nodes"])
    # wall_time
    if rec.get("wall_time"):
        rec["wall_time"] = hms_to_seconds(rec["wall_time"])

    return rec


def load_data(local_dataset_path, **kwargs):
    config = kwargs.get("config")
    root = Path(local_dataset_path[0])
    day = kwargs.get("start")
    fp = root / "torque_logs" / day
    filter_str = kwargs.get("filter")
    debug = kwargs.get("debug")

    jobs_raw = []

    # parse file
    for line in fp.open("rt", errors="ignore"):
        if "jobname" not in line.lower():
            continue
        rec = _parse_line(line)

        # skip if missing times
        if not (rec.get("start_time") and rec.get("end_time")):
            continue

        # ints
        st = int(rec["start_time"])
        et = int(rec["end_time"])
        sub = int(rec.get("submit_time", st))

        duration = et - st if et >= st else 0
        nr = int(rec.get("nodes_required"))
        int(rec.get("cpu_cores_required"))

        jid = rec.get("id")
        trace_quanta = config.get("TRACE_QUANTA")

        job_d = job_dict(
            nodes_required=nr,
            name=rec.get("name"),
            account=rec.get("account"),
            # cpu_trace=[0]*nr*nc,     # placeholder trace
            # gpu_trace=[0]*nr*0,      # Blue Waters has no GPUs
            cpu_trace=0,
            gpu_trace=0,
            nrx_trace=[],
            ntx_trace=[],
            end_state="UNKNOWN",
            scheduled_nodes=rec.get("scheduled_nodes"),
            id=jid,
            priority=0,
            submit_time=sub,
            time_limit=int(rec.get("wall_time")),
            start_time=st,
            end_time=et,
            expected_run_time=duration,
            current_run_time=0,
            trace_time=sub,
            trace_start_time=st,
            trace_end_time=et,
            trace_quanta=trace_quanta,
        )
        jobs_raw.append(job_d)

    # jobs_raw = list of dicts with absolute epoch times (as ints), e.g.:
    # {'id': '6335144.bw', 'name': '...', 'account': '...', 'scheduled_nodes': [26742, ...],
    #  'nodes_required': 4, 'cpu_cores_required': 32, 'submit_time': 1489798453,
    #  'start_time': 1489799118, 'end_time': 1489813275}

    # Gather global filters once
    all_nodes = set()
    abs_starts = []
    abs_ends = []

    for r in jobs_raw:
        if r.get("scheduled_nodes"):
            all_nodes.update(r["scheduled_nodes"])
        abs_starts.append(int(r["start_time"]))
        abs_ends.append(int(r["end_time"]))
    if not all_nodes or not abs_starts:
        return [], 0, 0

    global_tmin = min(abs_starts)
    global_tmax = max(abs_ends)

    # Confirm the correct 0-based indices for ipogif0_* from the HEADER
    # tx_idx = 15 # for the original file
    # rx_idx = 16
    tx_idx = 2  # for a downselected file with just four columns: [timestamp, node, tx, rx] - for faster loading
    rx_idx = 3

    # Build once (chunk if files are huge)
    sampler_df = build_sampler_df(root, day, all_nodes, global_tmin, global_tmax, tx_idx, rx_idx, chunksize=None)
    # Optional speed-ups:
    # sampler_df.set_index(["nid"], inplace=True)  # if you want .loc fast path per node

    # Option 1: take indices from kwargs (0-based). Option 2: keep your quick defaults.

    Path(local_dataset_path[0] if isinstance(local_dataset_path, (list, tuple)) else local_dataset_path)

    bin_s = config.get("TRACE_QUANTA")
    jobs = []

    for r in jobs_raw:  # Is this intended? We go throught the 'raw' jobs_dicts that were creeated above?
        st_abs = int(r["start_time"])
        et_abs = int(r["end_time"])
        nodes = r.get("scheduled_nodes") or []
        jid = r["id"]

        # Filter by nodes, sum positive deltas
        dfj = sampler_df[sampler_df["nid"].isin(nodes)]

        # Print first 10 rows (node, tx, rx)
        if debug:
            print(dfj[["nid", "tx", "rx"]].head(10))

        total_tx = int(dfj["tx"].sum()) if not dfj.empty else 0
        total_rx = int(dfj["rx"].sum()) if not dfj.empty else 0

        nodes_required = r.get("nodes_required")

        avg_tx_per_node = total_tx / nodes_required if nodes_required > 0 else 0
        avg_rx_per_node = total_rx / nodes_required if nodes_required > 0 else 0

        # Smear totals evenly across bins (simple first pass)
        duration = max(1, et_abs - st_abs)
        samples = max(1, math.ceil(duration / bin_s))
        ntx, nrx = throughput_traces(avg_tx_per_node, avg_rx_per_node, samples)

        job_d = job_dict(
            nodes_required=nodes_required,
            name=r.get("name"),
            account=r.get("account", "unknown"),
            cpu_trace=0,
            gpu_trace=0,
            nrx_trace=nrx,
            ntx_trace=ntx,
            end_state="UNKNOWN",
            scheduled_nodes=nodes,
            id=jid,
            priority=0,
            submit_time=int(r["submit_time"]),
            time_limit=int(r["time_limit"]),
            start_time=st_abs,
            end_time=et_abs,
            expected_run_time=et_abs - st_abs,
            current_run_time=0,
            trace_time=st_abs,
            trace_start_time=st_abs,
            trace_end_time=st_abs + samples * bin_s,
            trace_quanta=bin_s,
            trace_missing_values=False,
        )

        if filter_str:
            traffic = (avg_tx_per_node + avg_rx_per_node) / 2.
            keep_jobs = eval(filter_str)
            print(job_d["id"], filter_str, traffic, keep_jobs)
        else:
            keep_jobs = True

        if keep_jobs:
            jobs.append(Job(job_d))

    # Normalize times so first start = 0
    t0 = min((j.start_time for j in jobs), default=0)
    for j in jobs:
        j.submit_time -= t0
        j.start_time -= t0
        j.end_time -= t0
        j.trace_time -= t0
        j.trace_start_time -= t0
        j.trace_end_time -= t0

    # pprint(jobs)

    if debug:
        pprint(jobs)

    telemetry_start = 0
    telemetry_end = max((j.end_time for j in jobs), default=0)

    return WorkloadData(
        jobs=jobs,
        telemetry_start=telemetry_start, telemetry_end=telemetry_end,
        start_date=datetime.fromtimestamp(t0, timezone.utc),
    )
