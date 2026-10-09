"""
Back-test of start-time forecasts: record what the RAPS twin and Slurm predict, later compare
with when jobs actually started.

``record`` takes a snapshot every ``--every`` minutes and appends one row per eligible pending job
to ``<dir>/snapshots-YYYY-MM-DD.jsonl``:

- ``twin``: RAPS twin start (scheduler plans with limits, jobs end at their history-rule forecast)
- ``twin_limit``: the same simulation with every job running to its time limit (an ablation:
  how much of the twin's skill comes from the runtime forecast)
- ``slurm``: Slurm's own backfill estimate (squeue %S), when it has one

Rows hold job ids, sizes, limits, priorities and times only, no user or job names. Running jobs
get a row too (``kind: running``) with the forecast end, so runtime forecasts can be checked.

``evaluate`` looks up actual starts and ends with ``sacct`` and reports errors by how far ahead
the prediction was made and by job size.

    python -m raps.live.backtest record --dir $D/backtest --every 10
    python -m raps.live.backtest evaluate --dir $D/backtest
"""
import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from raps.live.analysis import eligible
from raps.live.forecast import Forecaster
from raps.live.slurm import SEP, ClusterState, Slurm, parse_slurm_time, run
from raps.live.twin import simulate

TFMT = "%Y-%m-%dT%H:%M:%S"


def _iso(dt):
    return dt.strftime(TFMT) if dt else None


class NoForecast:
    def predict(self, job):
        return None


def snapshot(slurm, forecaster, partition_nodes):
    now = datetime.now()
    state = ClusterState(user=slurm.user, jobs=slurm.squeue(), node_states=slurm.node_states(),
                         forecast=forecaster.refresh())
    twin = simulate(state, partition_nodes, now=now)
    plain = simulate(replace(state, forecast=NoForecast()), partition_nodes, now=now)
    rows = []
    for j in state.jobs:
        p = state.predict(j)
        base = {"t": _iso(now), "job": j.id, "nodes": j.nodes, "limit": j.limit, "partition": j.partition,
                "fcst_runtime": round(p.runtime) if p else None, "basis": p.basis if p else None}
        if eligible(j):
            rows.append({**base, "kind": "pending", "priority": j.priority, "submit": _iso(j.submit),
                         "reason": j.reason, "twin": _iso(twin.starts.get(j.id)),
                         "twin_limit": _iso(plain.starts.get(j.id)),
                         # an estimate in the past means Slurm expects to start it now
                         "slurm": _iso(max(j.start, now)) if j.start else None})
        elif j.running:
            rows.append({**base, "kind": "running", "start": _iso(j.start), "elapsed": j.elapsed,
                         "twin_end": _iso(twin.ends.get(j.id))})
    return now, rows, twin


def record(args):
    os.umask(0o077)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    slurm = Slurm()
    forecaster = Forecaster(days=args.forecast_days)
    partition_nodes = slurm.partition_nodes()
    while True:
        t0 = time.monotonic()
        try:
            now, rows, twin = snapshot(slurm, forecaster, partition_nodes)
            with open(out / f"snapshots-{now:%Y-%m-%d}.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            pend = sum(r["kind"] == "pending" for r in rows)
            print(f"{now:%Y-%m-%d %H:%M:%S} {pend} pending, {len(rows) - pend} running, "
                  f"twin {twin.seconds:.2f}s, took {time.monotonic() - t0:.1f}s", flush=True)
        except Exception:
            print(f"{datetime.now():%H:%M:%S} snapshot failed:\n{traceback.format_exc(limit=3)}", flush=True)
        if args.once:
            return
        time.sleep(max(args.every * 60 - (time.monotonic() - t0), 5))


# ---------------------------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------------------------


def actuals(since, until):
    """job id -> (start, end) for every job that started in [since, until] (sacct -a)."""
    out = {}
    text = run(["sacct", "-a", "-X", "-n", "-P", f"--delimiter={SEP}", "-S", since.strftime(TFMT),
                "-E", until.strftime(TFMT), "-o", "JobID,Start,End"], timeout=600)
    for line in text.splitlines():
        f = line.split(SEP)
        if len(f) == 3:
            start = parse_slurm_time(f[1])
            if start is not None:
                out[f[0]] = (start, parse_slurm_time(f[2]))
    return out


LEAD_BINS = [(0, 1), (1, 6), (6, 24), (24, 1e9)]       # hours between snapshot and actual start
SIZE_BINS = [(1, 15), (16, 127), (128, 1023), (1024, 10 ** 9)]
PREDICTORS = ("twin", "twin_limit", "slurm")


def _bin_label(lo, hi, unit=""):
    return f"≥{lo}{unit}" if hi >= 1e9 else f"{lo}–{hi}{unit}"


def evaluate(args):
    import numpy as np
    import pandas as pd

    files = sorted(Path(args.dir).glob("snapshots-*.jsonl"))
    if not files:
        sys.exit(f"no snapshots in {args.dir}")
    df = pd.concat((pd.read_json(f, lines=True, dtype=False) for f in files), ignore_index=True)
    for c in ("t", "submit", "start", "twin", "twin_limit", "slurm", "twin_end"):
        if c in df:
            df[c] = pd.to_datetime(df[c], format=TFMT, errors="coerce")
    t_first, t_last = df.t.min(), df.t.max()
    now = datetime.now()
    act = actuals(t_first.to_pydatetime() - timedelta(days=7), now)
    df["actual_start"] = df.job.map(lambda j: act.get(j, (None, None))[0])
    df["actual_end"] = df.job.map(lambda j: act.get(j, (None, None))[1])
    df["actual_start"] = pd.to_datetime(df.actual_start)
    df["actual_end"] = pd.to_datetime(df.actual_end)

    lines = [f"# Start-time back-test ({t_first:%Y-%m-%d %H:%M} to {t_last:%Y-%m-%d %H:%M}, "
             f"{df.t.nunique()} snapshots)", ""]
    pend = df[df.kind == "pending"].copy()
    # Only rows whose job has started can be scored (jobs still waiting are censored)
    started = pend.actual_start.notna()
    lines.append(f"{len(pend):,} pending-job rows ({pend.job.nunique():,} jobs); {started.mean() * 100:.1f}% of "
                 "rows have started since (the rest are excluded, so long waits are under-represented "
                 "until the recording has run for a while).")
    lines.append("")
    pend = pend[started]
    pend["lead_h"] = (pend.actual_start - pend.t).dt.total_seconds() / 3600
    pend = pend[pend.lead_h >= 0]
    for p in PREDICTORS:
        pend[f"err_{p}"] = (pend[p] - pend.actual_start).dt.total_seconds() / 3600

    def table(groups, title):
        out = [f"## {title}", "",
               "| group | rows | " + " | ".join(f"{p} median abs err (h) | {p} within ±25% of wait" for p in PREDICTORS)
               + " |", "|---|---|" + "---|---|" * len(PREDICTORS)]
        for label, g in groups:
            cells = [label, f"{len(g):,}"]
            for p in PREDICTORS:
                e = g[f"err_{p}"].dropna()
                if len(e) == 0:
                    cells += ["—", "—"]
                    continue
                within = (e.abs() <= 0.25 * g.loc[e.index, "lead_h"].clip(lower=0.25)).mean()
                cells += [f"{e.abs().median():.2f} (n={len(e):,})", f"{within * 100:.0f}%"]
            out.append("| " + " | ".join(cells) + " |")
        return out + [""]

    # Compare like with like: rows where every predictor (Slurm included) has an estimate
    both = pend.dropna(subset=[f"err_{p}" for p in PREDICTORS])
    lines += table([("all rows", pend), ("rows with all three estimates", both)], "Overall")
    lines += table([(_bin_label(lo, hi, " h"), pend[(pend.lead_h >= lo) & (pend.lead_h < hi)])
                    for lo, hi in LEAD_BINS], "By actual wait from the snapshot")
    lines += table([(_bin_label(lo, hi, " nodes"), pend[(pend.nodes >= lo) & (pend.nodes <= hi)])
                    for lo, hi in SIZE_BINS], "By job size")
    lines.append("Bias (median signed error, h; negative = predicted too early): " + ", ".join(
        f"{p} {pend[f'err_{p}'].median():+.2f}" for p in PREDICTORS))
    lines.append("")

    run_ = df[(df.kind == "running") & df.actual_end.notna() & df.twin_end.notna()].copy()
    if len(run_):
        run_["err_fcst"] = (run_.twin_end - run_.actual_end).dt.total_seconds() / 3600
        run_["err_limit"] = ((run_.start + pd.to_timedelta(run_.limit, unit="s")) - run_.actual_end
                             ).dt.total_seconds() / 3600
        lines += ["## End of running jobs", "",
                  f"{len(run_):,} running-job rows: median abs error {run_.err_fcst.abs().median():.2f} h "
                  f"with the forecast, {run_.err_limit.abs().median():.2f} h assuming the time limit; "
                  "node-weighted: "
                  f"{np.average(run_.err_fcst.abs(), weights=run_.nodes):.2f} h vs "
                  f"{np.average(run_.err_limit.abs(), weights=run_.nodes):.2f} h.", ""]
    report = "\n".join(lines)
    print(report)
    (Path(args.dir) / "report.md").write_text(report)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m raps.live.backtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--dir", required=True)
    r.add_argument("--every", type=float, default=10, help="Minutes between snapshots")
    r.add_argument("--forecast-days", type=int, default=30)
    r.add_argument("--once", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--dir", required=True)
    a = p.parse_args(argv)
    record(a) if a.cmd == "record" else evaluate(a)


if __name__ == "__main__":
    main()
