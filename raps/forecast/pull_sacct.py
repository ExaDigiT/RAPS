"""
Pull Frontier job accounting (all users) from sacct into weekly Parquet files.

Raw identifiers (user, job name, account, workdir) are kept: the output directory must stay
private and on OLCF systems. Weeks already written are skipped, so the pull can resume.

    python -m raps.forecast.pull_sacct --start 2024-01-01 --out $PROJWORK/stf053/exadigit/forecast-data/sacct
"""
import argparse
import io
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

FIELDS = ["JobIDRaw", "User", "Account", "JobName", "Partition", "QOS", "NNodes", "TimelimitRaw",
          "ElapsedRaw", "Submit", "Start", "End", "State", "ExitCode", "ConsumedEnergyRaw", "Priority",
          "WorkDir"]
SEP = "\x1f"


def pull_week(start, end):
    cmd = ["sacct", "-a", "-X", "-n", "-P", f"--delimiter={SEP}", "-S", start.isoformat(),
           "-E", end.isoformat(), "-o", ",".join(FIELDS)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True).stdout
    df = pd.read_csv(io.StringIO(out), sep=SEP, names=FIELDS, header=None, dtype=str,
                     keep_default_na=False, quoting=3, engine="python")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pause", type=float, default=2.0, help="seconds between queries, to go easy on slurmdbd")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    while d < end:
        e = min(d + timedelta(days=7), end)
        f = a.out / f"{d.isoformat()}.parquet"
        if not f.exists() or e == end:  # the current partial week is always refreshed
            t = time.time()
            df = pull_week(d, e)
            df.to_parquet(f, index=False)
            print(f"{d} -> {e}: {len(df):6d} rows  {time.time() - t:4.1f}s", flush=True)
            time.sleep(a.pause)
        d = e


if __name__ == "__main__":
    main()
