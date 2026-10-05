"""Retarget a RAPS job snapshot from one system to another, smaller one.

    python scripts/retarget_snapshot.py SNAPSHOT.npz --source frontier --target lux -o OUT.npz

A snapshot (snapshot.npz in any RAPS output directory) stores each job's CPU and
GPU traces as busy-device counts per node (utilization x devices per node) and
its node count on the source system. Replaying it unchanged on a system with
different nodes misreads both. This script, per job:

- scales cpu_trace and gpu_trace by target/source CPUs and GPUs per node, so
  each job keeps its utilization fraction (e.g. Frontier 4 GPUs -> Lux 8: x2)
- caps nodes_required at the target's available nodes (--oversize clamp), or
  drops jobs that do not fit (--oversize drop); smaller jobs keep their size,
  unlike `raps run --scale`, which redraws every job's size at random
- clears scheduled_nodes, since source placements do not exist on the target
- raises time_limit to the recorded run time where a job overran it (Slurm's
  kill grace: TIMEOUT jobs run a few seconds past their limit), since RAPS
  aborts on a job exceeding its limit once it schedules jobs itself

Replay the result with RAPS's own scheduler, since recorded start times and
placements no longer apply:

    raps run --system lux -f OUT.npz --policy fcfs --backfill firstfit -t 2h
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, if raps is not installed
from raps.system_config import get_system_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("snapshot")
    ap.add_argument("--source", required=True, help="system the snapshot was made on")
    ap.add_argument("--target", required=True, help="system to replay on")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--oversize", choices=["clamp", "drop"], default="clamp",
                    help="what to do with jobs larger than the target (default: clamp)")
    args = ap.parse_args()

    src = get_system_config(args.source).get_legacy()
    tgt = get_system_config(args.target).get_legacy()
    gpu_ratio = tgt["GPUS_PER_NODE"] / src["GPUS_PER_NODE"]
    cpu_ratio = tgt["CPUS_PER_NODE"] / src["CPUS_PER_NODE"]
    max_nodes = tgt["AVAILABLE_NODES"]

    snap = np.load(args.snapshot, allow_pickle=True)
    jobs, kept, clamped, overran = snap["jobs"], [], 0, 0
    for job in jobs:
        job = dict(job)
        if job["nodes_required"] > max_nodes:
            if args.oversize == "drop":
                continue
            job["nodes_required"] = max_nodes
            clamped += 1
        for key, ratio, cap in (("cpu_trace", cpu_ratio, tgt["CPUS_PER_NODE"]),
                                ("gpu_trace", gpu_ratio, tgt["GPUS_PER_NODE"])):
            trace = job.get(key)
            if trace is not None:
                job[key] = np.clip(np.asarray(trace, dtype=float) * ratio, 0, cap)
        if job["expected_run_time"] > job["time_limit"]:
            job["time_limit"] = job["expected_run_time"]
            overran += 1
        job["scheduled_nodes"] = None
        kept.append(job)

    out = {k: snap[k] for k in snap.files if k != "jobs"}
    out["jobs"] = np.array(kept, dtype=object)
    np.savez(args.output, **out)
    print(f"{len(jobs)} jobs from {args.source} -> {len(kept)} on {args.target} "
          f"({clamped} clamped to {max_nodes} nodes, {len(jobs) - len(kept)} dropped, "
          f"{overran} time limits raised to the recorded run time); "
          f"traces x{cpu_ratio:g} CPU, x{gpu_ratio:g} GPU -> {args.output}")


if __name__ == "__main__":
    main()
