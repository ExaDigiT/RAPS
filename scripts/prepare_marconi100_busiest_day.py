#!/usr/bin/env python3
"""
Filter the full PM100 (Marconi100) job trace down to just the jobs that
overlap the busiest single day in the corpus (2020-10-10, see
experiments/marconi100-busiest-day.yaml), so a 24h replay doesn't have to
pay RAPS's per-job trace-processing cost (a slow Python loop in
raps/dataloaders/marconi100.py::load_data_from_df) for all ~231k jobs in
the full 160-day corpus when only ~1k of them are ever visible in the
simulated window.

A job is kept if its recorded [start_time, end_time) interval overlaps the
target day at all (not just jobs that start within it), so jobs already
running at the start of the window still occupy nodes correctly -- this
matches the interval-overlap node-hours accounting used to find the
busiest day in the first place.

Usage:
    python scripts/prepare_marconi100_busiest_day.py [output_path]

Produces /opt/data/hpc/marconi100/job_table_busiest_day.parquet by
default. Replay with:
    raps run experiments/marconi100-busiest-day.yaml
(after pointing its `replay:` entry at the produced file).
"""
import sys
from pathlib import Path

import pandas as pd

SOURCE = Path("/opt/data/hpc/marconi100/job_table.parquet")
DAY_START = pd.Timestamp("2020-10-10T00:00:00", tz="UTC")
DAY_END = pd.Timestamp("2020-10-11T00:00:00", tz="UTC")


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/data/hpc/marconi100/job_table_busiest_day.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SOURCE, engine="pyarrow")
    mask = (df["start_time"] < DAY_END) & (df["end_time"] > DAY_START)
    subset = df[mask].reset_index(drop=True)

    print(f"Source: {len(df)} jobs")
    print(f"Overlapping {DAY_START.date()}: {len(subset)} jobs "
          f"({(subset['start_time'] >= DAY_START).sum()} start within the window)")
    print(f"Subset span: {subset['start_time'].min()} .. {subset['end_time'].max()}")

    subset.to_parquet(out_path, engine="pyarrow")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
