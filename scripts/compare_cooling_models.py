"""Compare the cooling outputs of two RAPS runs, e.g. the FMU against the ML surrogate.

Usage:

    python scripts/compare_cooling_models.py RUN_A RUN_B [--skip 2h] [--plot out.png]

RUN_A and RUN_B are RAPS output directories (each holding cooling_model.parquet),
typically produced with the same --seed and --time-delta, one with
--cooling-model fmu and one with --cooling-model surrogate. --skip drops the
start of both runs, e.g. while the FMU settles from its cold initial state.
Reports, per output type, the mean absolute difference and mean bias (B - A)
over all CDUs, next to how much the output varies in run A.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS = ["T_prim_s_C", "T_prim_r_C", "T_sec_s_C", "T_sec_r_C", "V_flow_prim_GPM",
           "V_flow_sec_GPM", "p_prim_s_psig", "p_prim_r_psig", "p_sec_s_psig",
           "p_sec_r_psig", "W_flow_CDUP_kW"]
CDU_COL = re.compile(r"computeBlock\[(\d+)\]\.cdu\[1\]\.summary\.(\w+)$")


def parse_duration(s):
    units = {"s": 1, "m": 60, "h": 3600}
    return int(float(s[:-1]) * units[s[-1]]) if s[-1] in units else int(s)


def load(run_dir):
    df = pd.read_parquet(Path(run_dir) / "cooling_model.parquet")
    return df.groupby("time").last()


def cdu_cols(df, output):
    cols = [c for c in df.columns if (m := CDU_COL.search(c)) and m.group(2) == output]
    return sorted(cols, key=lambda c: int(CDU_COL.search(c).group(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--skip", default="0s", help="drop this much from the start (e.g. 2h)")
    ap.add_argument("--plot", help="write a figure of system-mean traces to this path")
    args = ap.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    t = a.index.intersection(b.index)
    t = t[t >= t.min() + parse_duration(args.skip)]
    if len(t) == 0:
        raise SystemExit("no overlapping timesteps after --skip")
    a, b = a.loc[t], b.loc[t]
    print(f"A = {args.run_a}\nB = {args.run_b}")
    print(f"compared {len(t)} timesteps, t = {t.min()}..{t.max()} s\n")

    rows = []
    for out in OUTPUTS:
        ca, cb = cdu_cols(a, out), cdu_cols(b, out)
        if not ca or ca != cb:
            continue
        xa, xb = a[ca].to_numpy(), b[cb].to_numpy()
        rows.append(dict(output=out, mean_A=xa.mean(), mean_B=xb.mean(),
                         MAE=np.abs(xb - xa).mean(), bias=(xb - xa).mean(),
                         std_A_time=xa.std(axis=0).mean(),
                         corr_time=np.nanmean([np.corrcoef(xa[:, i], xb[:, i])[0, 1]
                                               for i in range(xa.shape[1])])))
    res = pd.DataFrame(rows).set_index("output")
    with pd.option_context("display.float_format", "{:9.3f}".format):
        print(res.to_string())
    if "pue" in a and "pue" in b:
        print(f"\nPUE mean: A {a['pue'].mean():.3f}  B {b['pue'].mean():.3f} "
              "(the surrogate's PUE excludes central-plant power)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
        for ax, out in zip(axes.ravel(), res.index):
            ax.plot(t / 3600, a[cdu_cols(a, out)].mean(axis=1), label="A", lw=1.2)
            ax.plot(t / 3600, b[cdu_cols(b, out)].mean(axis=1), label="B", lw=1.2, ls="--")
            ax.set_title(out, fontsize=10)
        axes.ravel()[-1].axis("off")
        axes[0, 0].legend()
        for ax in axes[-1]:
            ax.set_xlabel("time (h)")
        fig.suptitle("System-mean cooling outputs: A (solid) vs B (dashed)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\nfigure: {args.plot}")


if __name__ == "__main__":
    main()
