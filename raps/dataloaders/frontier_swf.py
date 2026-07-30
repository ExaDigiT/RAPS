"""
Independent RAPS-native dataloader for the public Frontier workload trace
(Maheshwari, Klusacek, Suter, Brewer, "Workload Traces from the Frontier
Exascale Supercomputer", JSSPP 2026), in Standard Workload Format (SWF).

This is deliberately a separate implementation from REDI's SWF ingestion
pipeline (redi/domains/hpc/swf_loader.py in the REDI repo) -- it parses the
raw .swf files directly, without importing or reusing that code -- so it can
serve as an independent ground-truth loader when cross-validating REDI's
ingestion (agreement between two independent readings of the same raw file
is meaningful evidence; agreement with itself is not).

Source data (whitespace-delimited, `;`-commented header, 21 fields/job):
    /opt/data/hpc/frontier/frontier-2024.swf
    /opt/data/hpc/frontier/frontier-2025.swf

This is scheduler-scalar data only (submit/wait/run times, node counts) --
there are no per-job power/CPU/GPU utilization traces, so replay drives
job scheduling/queuing, not telemetry-based power modeling.

Usage:

    raps run --dataloader raps.dataloaders.frontier_swf --system frontier \\
        --replay /opt/data/hpc/frontier/frontier-2024.swf \\
                 /opt/data/hpc/frontier/frontier-2025.swf \\
        -t 1 --noui --no-cooling

Node-count caveat: SWF field 4 ("allocated_nodes") is documented as "number
of all allocated nodes/or number of all allocated CPU cores" and is
frequently inconsistent with Frontier's ~9,408-node size (values up to
589,824 observed in the 2024 file). The trailing site descriptor field 20
(e.g. "frontier:excl:2500x112") carries a node count N for which `used_GPUs`
(field 18, when populated) is evenly divisible essentially 100% of the time,
vs. 16-43% for the raw numeric field -- so N is the reliable node count and
is what this loader uses; the raw field 4/7 is ignored.
"""
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from raps.job import Job, job_dict
from raps.utils import WorkloadData

# SWF fields 0-20, in file order (both Frontier trace files carry all 21).
_SWF_COLUMNS = [
    "job_id", "submit_time_rel", "wait_time", "runtime", "nodes_field",
    "cpu_time_used", "ram_used_kb", "required_nodes_field", "walltime_limit",
    "ram_requested_kb", "job_status", "user_id", "group_id",
    "executable_number", "queue_id", "partition_id", "preceding_job",
    "think_time", "used_gpus", "soft_walltime", "descriptor",
]

# Trailing descriptor field, e.g. "frontier:excl:2500x112" -> node count 2500.
_DESCRIPTOR_RE = re.compile(r"^[^:]+:[^:]+:(\d+)x\d+$")

# Not present per-job in SWF; fixed at the same value REDI's ingestion config
# uses (examples/hpc/frontier_swf.yaml: raps_jobs.trace_quanta), so this is a
# shared pipeline parameter rather than something either side derives from data.
TRACE_QUANTA = 15


def _read_unix_start_time(path: Path) -> int:
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith(";"):
                break
            m = re.match(r"^;\s*UnixStartTime\s*:\s*(\d+)\s*$", line)
            if m:
                return int(m.group(1))
    raise ValueError(f"{path}: no '; UnixStartTime: <epoch>' header line found")


def _parse_one_file(path: Path) -> pd.DataFrame:
    unix_start_time = _read_unix_start_time(path)
    df = pd.read_csv(path, comment=";", sep=r"\s+", header=None,
                      names=_SWF_COLUMNS, engine="c")

    descriptor_n = pd.to_numeric(
        df["descriptor"].astype(str).str.extract(_DESCRIPTOR_RE)[0], errors="coerce")
    unparseable = descriptor_n.isna()
    if unparseable.any():
        # Fall back to the raw numeric field only where the descriptor doesn't
        # parse (not expected to occur on the Frontier trace files).
        descriptor_n = descriptor_n.where(~unparseable, df["nodes_field"])
    df["nodes_required"] = descriptor_n.astype(np.int64)

    df["submit_time_abs"] = unix_start_time + df["submit_time_rel"]
    df["start_time_abs"] = df["submit_time_abs"] + df["wait_time"]
    df["end_time_abs"] = df["start_time_abs"] + df["runtime"]
    df["source_file"] = path.name
    return df


def load_data(files, **kwargs):
    """
    Parse Frontier public SWF trace file(s) into RAPS Job objects.

    Args:
        files (list[str | Path]): one or more .swf files (e.g. frontier-2024.swf,
            frontier-2025.swf).

    Returns:
        WorkloadData

    Note: job_id is only unique within a single SWF file. When multiple files
    are given, ids are namespaced as "<filename>:<job_id>" -- matching REDI's
    swf_loader.py convention for the same reason (cross-file id collisions),
    which also happens to keep ids aligned 1:1 for cross-validation.
    """
    paths = [Path(f) for f in files]
    dfs = [_parse_one_file(p) for p in paths]
    df = pd.concat(dfs, axis=0, ignore_index=True) if len(dfs) > 1 else dfs[0]

    if len(paths) > 1:
        df["job_id"] = df["source_file"].astype(str) + ":" + df["job_id"].astype(str)

    # Relative-to-telemetry-start timestamps, per RAPS convention (e.g.
    # raps/dataloaders/frontier.py, lassen.py): differences (wait_time,
    # run_time) are shift-invariant, so this choice doesn't affect validation,
    # but it keeps this loader consistent/reusable for actual replay too.
    telemetry_start_ts = df["submit_time_abs"].min()
    df["submit_time"] = df["submit_time_abs"] - telemetry_start_ts
    df["start_time"] = df["start_time_abs"] - telemetry_start_ts
    df["end_time"] = df["end_time_abs"] - telemetry_start_ts

    jobs = []
    for row in df.itertuples(index=False):
        used_gpus = int(row.used_gpus) if row.used_gpus != -1 else 0
        walltime_limit = float(row.walltime_limit) if row.walltime_limit != -1 else 0.0
        info = job_dict(
            id=str(row.job_id),
            name=f"frontier-swf-{row.job_id}",
            account=str(row.user_id),
            nodes_required=int(row.nodes_required),
            gpu_units_required=used_gpus,
            scheduled_nodes=[],
            cpu_trace=[], gpu_trace=[], ntx_trace=[], nrx_trace=[],
            submit_time=float(row.submit_time),
            start_time=float(row.start_time),
            end_time=float(row.end_time),
            time_limit=walltime_limit,
            expected_run_time=float(row.runtime),
            trace_quanta=TRACE_QUANTA,
        )
        jobs.append(Job(info))

    telemetry_end = int(df["end_time"].max())
    start_date = datetime.fromtimestamp(int(telemetry_start_ts), tz=timezone.utc)

    return WorkloadData(
        jobs=jobs,
        telemetry_start=0,
        telemetry_end=telemetry_end,
        start_date=start_date,
    )


def node_index_to_name(index: int, config: dict):
    """SWF carries no node identities; index is not meaningful, kept for interface parity."""
    return f"node{index:05d}"


def cdu_index_to_name(index: int, config: dict):
    return f"cdu{index:02d}"


def cdu_pos(index: int, config: dict) -> tuple[int, int]:
    return (0, index)
