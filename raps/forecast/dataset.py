"""
Build the per-job training table from the weekly sacct pulls.

One row per job that actually ran (Start set, ElapsedRaw > 0) and has a time limit. Targets:

- ``log_runtime``: log1p(ElapsedRaw seconds)
- ``ratio``: runtime / time limit, clipped to [0, 1.2]
- ``node_power``: ConsumedEnergyRaw / ElapsedRaw / NNodes in watts (NaN where energy is missing;
  Frontier's accounting starts recording energy mid-June 2024)

History features use only jobs that *ended before this job was submitted*, which is exactly what
would be known at submit time, so there is no leakage from the future. They are computed per
(user, job name), per user, and per account: number of earlier jobs, mean ratio / log runtime of
the last 5 and of all of them, and the most recent ratio.

    python -m raps.forecast.dataset --sacct $D/sacct --out $D/jobs.parquet
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

RAN_STATES = ("COMPLETED", "TIMEOUT", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL")
HIST_KEYS = {"uj": ["user", "name"], "u": ["user"], "a": ["account"]}
LAST_N = 5
TFMT = "%Y-%m-%dT%H:%M:%S"
POWER_MIN_W, POWER_MAX_W = 50, 5000  # plausible mean watts per Frontier node


def load_raw(sacct_dir):
    files = sorted(glob.glob(str(Path(sacct_dir) / "*.parquet")))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    # A job overlapping several weekly windows appears once per window; the latest pull wins
    return df.drop_duplicates("JobIDRaw", keep="last")


def clean(df):
    out = pd.DataFrame({
        "job_id": df.JobIDRaw.astype(str),
        "user": df.User, "account": df.Account, "name": df.JobName,
        "partition": df.Partition, "qos": df.QOS, "workdir": df.WorkDir,
        "state": df.State.str.split().str[0],
        "nodes": pd.to_numeric(df.NNodes, errors="coerce"),
        "limit": pd.to_numeric(df.TimelimitRaw, errors="coerce") * 60,
        "runtime": pd.to_numeric(df.ElapsedRaw, errors="coerce"),
        "energy": pd.to_numeric(df.ConsumedEnergyRaw, errors="coerce"),
        "priority": pd.to_numeric(df.Priority, errors="coerce"),
        "submit": pd.to_datetime(df.Submit, format=TFMT, errors="coerce"),
        "start": pd.to_datetime(df.Start, format=TFMT, errors="coerce"),
        "end": pd.to_datetime(df.End, format=TFMT, errors="coerce"),
    })
    ok = (out.state.isin(RAN_STATES) & out.start.notna() & out.end.notna() & out.submit.notna()
          & (out.runtime > 0) & (out.limit > 0) & (out.nodes > 0))
    out = out[ok].copy()
    out["log_runtime"] = np.log1p(out.runtime)
    out["ratio"] = (out.runtime / out.limit).clip(0, 1.2)
    out["log_limit"] = np.log1p(out.limit)
    out["log_nodes"] = np.log2(out.nodes)
    power = out.energy / out.runtime / out.nodes
    # Implausible readings (energy counter glitches give values up to ~1e9 W) are treated as missing
    out["node_power"] = power.where((out.energy > 0) & power.between(POWER_MIN_W, POWER_MAX_W))
    out["wait"] = (out.start - out.submit).dt.total_seconds().clip(lower=0)
    out["submit_hour"] = out.submit.dt.hour + out.submit.dt.minute / 60
    out["submit_dow"] = out.submit.dt.dayofweek
    return out.sort_values("submit").reset_index(drop=True)


def _history(df, keys, prefix):
    """
    For each job: stats over earlier jobs in the same group that ended before it was submitted.
    Within a group, jobs are ordered by end time; ``searchsorted`` on submit time gives how many
    had finished, and cumulative sums give means over all / the last LAST_N of them.
    """
    n = len(df)
    cols = {f"{prefix}_n": np.zeros(n), f"{prefix}_ratio_last": np.full(n, np.nan),
            f"{prefix}_ratio_mean5": np.full(n, np.nan), f"{prefix}_ratio_mean": np.full(n, np.nan),
            f"{prefix}_logrt_mean5": np.full(n, np.nan), f"{prefix}_power_mean5": np.full(n, np.nan)}
    end_ns = df.end.values.astype("datetime64[ns]").astype(np.int64)
    sub_ns = df.submit.values.astype("datetime64[ns]").astype(np.int64)
    ratio, logrt, power = df.ratio.values, df.log_runtime.values, df.node_power.values
    for idx in df.groupby(keys, sort=False).indices.values():
        if len(idx) == 1:
            continue
        order = idx[np.argsort(end_ns[idx], kind="stable")]
        ends = end_ns[order]
        k = np.searchsorted(ends, sub_ns[idx], side="left")  # jobs in group finished before submit
        has = k > 0
        if not has.any():
            continue
        r_cs = np.concatenate(([0.0], np.cumsum(ratio[order])))
        l_cs = np.concatenate(([0.0], np.cumsum(logrt[order])))
        p = power[order]
        p_ok = ~np.isnan(p)
        p_cs = np.concatenate(([0.0], np.cumsum(np.where(p_ok, p, 0.0))))
        p_cnt = np.concatenate(([0], np.cumsum(p_ok)))
        lo = np.maximum(k - LAST_N, 0)
        w = np.maximum(k - lo, 1)
        cols[f"{prefix}_n"][idx] = k
        sel, ks, los, ws = idx[has], k[has], lo[has], w[has]
        cols[f"{prefix}_ratio_last"][sel] = ratio[order][ks - 1]
        cols[f"{prefix}_ratio_mean5"][sel] = (r_cs[ks] - r_cs[los]) / ws
        cols[f"{prefix}_ratio_mean"][sel] = r_cs[ks] / ks
        cols[f"{prefix}_logrt_mean5"][sel] = (l_cs[ks] - l_cs[los]) / ws
        pc = p_cnt[ks] - p_cnt[los]
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"{prefix}_power_mean5"][sel] = np.where(pc > 0, (p_cs[ks] - p_cs[los]) / pc, np.nan)
    return pd.DataFrame(cols, index=df.index)


def add_history(df):
    parts = [_history(df, keys, prefix) for prefix, keys in HIST_KEYS.items()]
    df = pd.concat([df] + parts, axis=1)
    df["log_uj_n"] = np.log1p(df.uj_n)
    df["log_u_n"] = np.log1p(df.u_n)
    # The history rule's own estimate (most specific group with history), as one feature
    r = df.uj_ratio_mean5.fillna(df.u_ratio_mean5).fillna(df.a_ratio_mean5)
    df["hist_log_pred"] = np.log1p(r.clip(1e-4, 1.2) * df.limit)
    return df


def build(sacct_dir):
    return add_history(clean(load_raw(sacct_dir)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sacct", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    df = build(a.sacct)
    df.to_parquet(a.out, index=False)
    print(f"{len(df):,} jobs, {df.user.nunique():,} users, {df.submit.min()} .. {df.submit.max()} -> {a.out}")


if __name__ == "__main__":
    main()
