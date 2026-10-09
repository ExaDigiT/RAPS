"""
Export the live queue as a RAPS replay, to run into the future with ``raps run``.

``export`` writes a folder under ``~/raps-forecast-runs`` (or $RAPS_LIVE_RUNS), named by local time and
also linked as ``latest``, with
- ``jobs.npz``: the same snapshot format ``Telemetry.save_snapshot`` writes. Running jobs are placed
  on their real nodes with the start time they have, pending jobs carry their submit time, priority
  and time limit.
- ``experiment.yaml``: ``raps run experiment.yaml`` runs it for ``hours`` (default 24).

There are no utilization traces for jobs that have not run. Each job gets one constant utilization
that makes the RAPS power model give the watts per node we expect from it: what a running job draws
now (Slurm's CurrentWatts), else the history rule's forecast, else the median of the running jobs.
Runtimes are the history rule's forecasts (the time limit where there is no history).

Not modelled: jobs submitted after the export, so the machine drains over the horizon.
"""
import json
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from raps.live.analysis import eligible, job_power
from raps.live.slurm import expand_hostlist, node_index

HOURS = 24


def _util_for_watts(watts, idle_w, max_w):
    """Fraction of a node's full CPU and GPU load that draws ``watts``. The model is linear in it."""
    return min(max((watts - idle_w) / (max_w - idle_w), 0.0), 1.0) if max_w > idle_w else 0.0


def build(state, system, hours=HOURS):
    """(job dicts, report) for the live queue in ``state``, as RAPS replay jobs. Time 0 is now."""
    from raps.job import Job, job_dict
    from raps.power import compute_node_power
    from raps.system_config import get_system_config

    cfg = get_system_config(system).get_legacy()
    cpus, gpus = cfg["CPUS_PER_NODE"], cfg["GPUS_PER_NODE"]
    idle_w = compute_node_power(0, 0, 0, cfg)[0]
    max_w = compute_node_power(cpus, gpus, 0, cfg)[0]  # the simulation models no network traffic
    total = cfg["TOTAL_NODES"]
    usable = set(range(total)) - set(cfg.get("DOWN_NODES") or ())  # includes the missing racks
    now = datetime.now()

    running = [j for j in state.jobs or [] if j.running and j.nodelist and j.limit]
    queued = [j for j in state.jobs or [] if eligible(j) and j.limit]
    measured = {j.id: w for j in running if (w := job_power(j, state.node_power))}
    typical = statistics.median(measured.values()) if measured else (idle_w + max_w) / 2

    def watts(j):
        p = state.predict(j)
        w = measured.get(j.id) or (p.power if p is not None and p.power else None) or typical
        return min(max(w, idle_w), max_w)

    def make(j, **times):
        u = _util_for_watts(watts(j), idle_w, max_w)
        d = job_dict(nodes_required=j.nodes, name=j.name, account=j.account, id=int(j.id.split("_")[0]), priority=j.priority,
                     cpu_trace=u * cpus, gpu_trace=u * gpus, ntx_trace=0, nrx_trace=0, time_limit=int(j.limit),
                     **times)
        return vars(Job(d))

    jobs, report = [], {"running": 0, "relocated": 0, "dropped": 0, "queued": 0, "too_big": 0}
    taken = set()
    pending_place = []
    # Slurm's nodes are not numbered like the RAPS layout (Frontier: 9,856 names up to 10495, against RAPS's
    # 9,600 slots with the missing rack as a hole), so the n-th node of the cluster becomes the n-th usable one
    ranked = sorted({(node_index(n) or 0) for n in state.node_states or {}})
    engine_id = dict(zip(ranked, sorted(usable)))
    for j in sorted(running, key=lambda j: -j.nodes):
        ids = [engine_id.get(node_index(n) or 0) for n in expand_hostlist(j.nodelist)]
        if len(ids) == j.nodes and None not in ids and len(set(ids)) == len(ids) and taken.isdisjoint(ids):
            taken.update(ids)
            pending_place.append((j, ids))
        else:
            pending_place.append((j, None))
    free = iter(sorted(usable - taken))
    for j, ids in pending_place:
        if ids is None:  # names that do not map onto the RAPS layout: put the job on free nodes
            ids = [int(i) for _, i in zip(range(j.nodes), free)]
            if len(ids) < j.nodes:
                report["dropped"] += 1
                continue
            taken.update(ids)
            report["relocated"] += 1
        p = state.predict(j)
        # Slurm keeps a job a while after its limit before it ends it; RAPS rejects running past the limit
        elapsed = int(min(j.elapsed, j.limit - 1))
        runtime = p.runtime if p is not None and p.runtime > elapsed else j.limit
        runtime = int(min(max(runtime, elapsed + 1), j.limit))
        start = -elapsed
        jobs.append(make(j, submit_time=min(start, int((j.submit - now).total_seconds())) if j.submit else start,
                         start_time=start, end_time=start + runtime, expected_run_time=runtime,
                         scheduled_nodes=ids, current_run_time=elapsed))
        report["running"] += 1
    capacity = len(usable)
    for j in queued:
        if j.nodes > capacity:
            report["too_big"] += 1
            continue
        p = state.predict(j)
        runtime = int(max(min(p.runtime if p is not None else j.limit, j.limit), 1))
        submit = int((j.submit - now).total_seconds()) if j.submit else 0
        jobs.append(make(j, submit_time=min(submit, 0), start_time=None, end_time=None, expected_run_time=runtime))
        report["queued"] += 1
    report.update(idle_w=round(idle_w), max_w=round(max_w), typical_w=round(typical), hours=hours, system=system)
    return jobs, report


def export(state, system, hours=HOURS, root=None, policy="priority", backfill="firstfit"):
    """Write the replay for the queue in ``state``. Returns (folder, experiment yaml path, report)."""
    jobs, report = build(state, system, hours)
    at = datetime.now(timezone.utc).replace(microsecond=0)
    root = Path(root or os.environ.get("RAPS_LIVE_RUNS") or Path.home() / "raps-forecast-runs")
    folder = root / f"{datetime.now():%Y-%m-%d_%H%M%S}"  # local time, easy to read; the yaml records UTC
    folder.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"  # always the newest export: ~/raps-forecast-runs/latest/experiment.yaml
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(folder.name)
    except OSError:
        pass
    from raps.telemetry import TelemetryArgs  # `raps run` reads the system name from the snapshot's args
    np.savez_compressed(folder / "jobs.npz", jobs=jobs, telemetry_start=0, telemetry_end=int(hours * 3600),
                        start_date=at, args=TelemetryArgs(system=system, live=True))
    yaml = folder / "experiment.yaml"
    yaml.write_text(f"""# The live queue at {at:%Y-%m-%d %H:%M:%S} UTC, run {hours} h into the future.
# Run with: raps run {yaml}
# Jobs: {report['running']} running, {report['queued']} queued; no jobs submitted after the export.
# Power per job is a constant utilization giving its measured or forecast watts per node.
system: {system}
replay:
  - {folder / 'jobs.npz'}
time: {hours}h
policy: {policy}
backfill: {backfill}
time_delta: 15s
# cooling: true     # needs the system's FMU
# noui: true        # no console, just the final statistics
""")
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    return folder, yaml, report
