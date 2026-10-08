"""
Compare per-job runtime and power predictors on Frontier sacct data, from simplest to richest.

Runtime ladder (target log1p(runtime)):
  limit        the job runs to its time limit (what Slurm's backfill assumes)
  history      mean ratio of the last 5 runs of the same user+job name, else user, else account
  gbm_request  LightGBM on what was requested only (limit, nodes, partition, qos, submit time)
  gbm_emb      ... plus the MiniLM identity embedding (no history)
  gbm_hist     ... request plus history features (no embedding)
  gbm_full     ... request plus history plus embedding

Power ladder (target W per node, jobs with energy accounting only): history, gbm_request, gbm_hist,
gbm_full.

Splits: chronological (train < --val-start <= val < --test-start <= test < --test-end) and a
held-out set of users (--holdout-frac of users by stable hash) who are dropped from train and val
entirely. Test metrics are reported for all test jobs, for held-out users, and for first runs
(no earlier run of that user+job name).

    python -m raps.forecast.evaluate --jobs $D/jobs.parquet --emb $D/embeddings.parquet --out $D/results
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from raps.forecast.embed import identity_string

REQUEST = ["log_limit", "log_nodes", "partition", "qos", "submit_hour", "submit_dow"]
HIST = [f"{p}_{s}" for p in ("uj", "u", "a")
        for s in ("ratio_last", "ratio_mean5", "ratio_mean", "logrt_mean5")] + ["log_uj_n", "log_u_n", "a_n", "hist_log_pred"]
POWER_HIST = ["uj_power_mean5", "u_power_mean5", "a_power_mean5"]
CATS = ["partition", "qos"]
EMB_DIMS = 32


def holdout_users(users, frac, seed="raps-forecast"):
    h = users.map(lambda u: int(hashlib.sha256(f"{seed}:{u}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    return h < frac


def add_embedding(df, emb_path, train_mask, dims=EMB_DIMS):
    """Join MiniLM vectors by identity string and reduce them with a PCA fit on training rows only."""
    from sklearn.decomposition import PCA
    emb = pd.read_parquet(emb_path)
    ecols = [c for c in emb.columns if c.startswith("e")]
    idx = pd.Index(emb.identity).get_indexer(identity_string(df))
    assert (idx >= 0).all(), "jobs without an embedding: rerun embed.py on this jobs table"
    X = emb[ecols].to_numpy(np.float32)[idx]
    pca = PCA(n_components=dims, random_state=0).fit(X[train_mask.to_numpy()][::5])
    Z = pca.transform(X)
    names = [f"emb{i}" for i in range(dims)]
    return pd.concat([df, pd.DataFrame(Z, columns=names, index=df.index)], axis=1), names, \
        float(pca.explained_variance_ratio_.sum())


def wquantile(x, w, q=0.5):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q * c[-1])])


def metrics(y_log, p_log, w=None):
    """
    y/p in log1p(seconds). R2 in log space plus errors people can read. With weights `w` (requested node-hours)
    every statistic is weighted, so jobs count by how much of the machine they ask for.
    """
    w = np.ones(len(y_log)) if w is None else np.asarray(w, float)
    y, p = np.expm1(y_log), np.expm1(p_log)
    ybar = np.average(y_log, weights=w)
    r2 = 1 - np.sum(w * (y_log - p_log) ** 2) / np.sum(w * (y_log - ybar) ** 2)
    rel = np.abs(p - y) / np.maximum(y, 60)
    return {"n": int(len(y)), "r2_log": float(r2), "medae_min": wquantile(np.abs(p - y), w) / 60,
            "median_rel_err": wquantile(rel, w), "within_25pct": float(np.average(rel <= 0.25, weights=w)),
            "under_pred": float(np.average(p < 0.8 * y, weights=w))}


def power_metrics(y, p, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ybar = np.average(y, weights=w)
    r2 = 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2)
    return {"n": int(len(y)), "r2": float(r2), "mae_w": float(np.average(np.abs(y - p), weights=w)),
            "median_rel_err": wquantile(np.abs(y - p) / np.maximum(y, 1), w)}


def train_weights(df, mask, how):
    """
    'none': every job counts once. 'user': each user's jobs share weight ~sqrt(their job count).
    'nodehours': requested node-hours (nodes x time limit), known at submit, so big jobs that shape
    the queue count in proportion to the machine they ask for.
    """
    if how == "none":
        return None
    if how == "nodehours":
        return (df.loc[mask, "nodes"] * df.loc[mask, "limit"] / 3600).to_numpy()
    counts = df.loc[mask, "user"].map(df.loc[mask, "user"].value_counts())
    return (1.0 / np.sqrt(counts)).to_numpy()


def fit_gbm(X, y, Xv, yv, threads, w=None, wv=None, seed=0):
    import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=threads, seed=seed, verbose=-1)
    dtr = lgb.Dataset(X, y, weight=w, categorical_feature=[c for c in CATS if c in X.columns], free_raw_data=False)
    dva = lgb.Dataset(Xv, yv, weight=wv, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def history_runtime(df, global_ratio):
    r = df.uj_ratio_mean5.fillna(df.u_ratio_mean5).fillna(df.a_ratio_mean5).fillna(global_ratio)
    return np.log1p(r.clip(1e-4, 1.2) * df.limit)


def history_power(df, global_power):
    return df.uj_power_mean5.fillna(df.u_power_mean5).fillna(df.a_power_mean5).fillna(global_power).to_numpy()


def subsets(test, held):
    return {"all": np.ones(len(test), bool), "heldout_users": held[test.index].to_numpy(),
            "first_run": (test.uj_n == 0).to_numpy()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--emb", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-start", default="2026-01-01")
    p.add_argument("--test-start", default="2026-04-01")
    p.add_argument("--test-end", default="2026-09-24")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--train-weight", choices=["none", "user", "nodehours"], default="none",
                   help="'user' down-weights prolific submitters; 'nodehours' weights by requested node-hours")
    p.add_argument("--sample", type=float, default=1.0, help="fraction of jobs to use (quick dev runs)")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.jobs)
    if a.sample < 1:
        df = df.sample(frac=a.sample, random_state=0)
    for c in CATS:
        df[c] = df[c].astype("category")
    held = holdout_users(df.user, a.holdout_frac)
    tr = (df.submit < a.val_start) & ~held
    va = (df.submit >= a.val_start) & (df.submit < a.test_start) & ~held
    te = (df.submit >= a.test_start) & (df.submit < a.test_end)
    emb_cols, pca_var = [], None
    if a.emb:
        df, emb_cols, pca_var = add_embedding(df, a.emb, tr)
    print(f"train {tr.sum():,}  val {va.sum():,}  test {te.sum():,} "
          f"(held-out users {held[te].sum():,}, first runs {(df[te].uj_n == 0).sum():,})", flush=True)

    test = df[te]
    subs = subsets(test, held)
    results = {"split": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                         "val_start": a.val_start, "test_start": a.test_start, "test_end": a.test_end,
                         "holdout_frac": a.holdout_frac, "pca_explained_var": pca_var},
               "train_weight": a.train_weight,
               "runtime": {}, "runtime_nh": {}, "power": {}, "power_nh": {}, "importance": {}}

    # ---- runtime --------------------------------------------------------------------------
    preds = {"limit": np.log1p(test.limit.to_numpy()),
             "history": history_runtime(test, df[tr].ratio.median()).to_numpy()}
    feats = {"gbm_request": REQUEST, "gbm_hist": REQUEST + HIST}
    if emb_cols:
        feats["gbm_emb"] = REQUEST + emb_cols
        feats["gbm_full"] = REQUEST + HIST + emb_cols
    for name, cols in feats.items():
        m = fit_gbm(df.loc[tr, cols], df.loc[tr, "log_runtime"], df.loc[va, cols], df.loc[va, "log_runtime"], a.threads,
                    train_weights(df, tr, a.train_weight), train_weights(df, va, a.train_weight))
        preds[name] = m.predict(test[cols], num_iteration=m.best_iteration)
        imp = pd.Series(m.feature_importance("gain"), index=cols)
        results["importance"][f"runtime/{name}"] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
        print(f"runtime {name}: {m.best_iteration} trees", flush=True)
    y = test.log_runtime.to_numpy()
    # Weight by *requested* node-hours, known at submit. Weighting by actual node-hours would favor
    # jobs that happened to run long and so reward predicting the time limit.
    nh = (test.nodes * test.limit / 3600).to_numpy()
    for name, pr in preds.items():
        results["runtime"][name] = {s: metrics(y[mask], pr[mask]) for s, mask in subs.items() if mask.any()}
        results["runtime_nh"][name] = {s: metrics(y[mask], pr[mask], nh[mask]) for s, mask in subs.items() if mask.any()}

    # ---- power ----------------------------------------------------------------------------
    has_p = df.node_power.notna()
    ptr, pva, pte = tr & has_p, va & has_p, te & has_p
    ptest = df[pte]
    psubs = subsets(ptest, held)
    ppred = {"history": history_power(ptest, df.loc[ptr, "node_power"].median())}
    pfeats = {"gbm_request": REQUEST, "gbm_hist": REQUEST + HIST + POWER_HIST}
    if emb_cols:
        pfeats["gbm_full"] = REQUEST + HIST + POWER_HIST + emb_cols
    for name, cols in pfeats.items():
        m = fit_gbm(df.loc[ptr, cols], df.loc[ptr, "node_power"], df.loc[pva, cols], df.loc[pva, "node_power"], a.threads,
                    train_weights(df, ptr, a.train_weight), train_weights(df, pva, a.train_weight))
        ppred[name] = m.predict(ptest[cols], num_iteration=m.best_iteration)
        print(f"power {name}: {m.best_iteration} trees", flush=True)
    yp = ptest.node_power.to_numpy()
    pnh = (ptest.nodes * ptest.limit / 3600).to_numpy()
    for name, pr in ppred.items():
        results["power"][name] = {s: power_metrics(yp[mask], pr[mask]) for s, mask in psubs.items() if mask.any()}
        results["power_nh"][name] = {s: power_metrics(yp[mask], pr[mask], pnh[mask]) for s, mask in psubs.items() if mask.any()}
    results["power_target"] = {"train_jobs": int(ptr.sum()), "test_jobs": int(pte.sum()),
                               "w_per_node_quantiles": df.loc[has_p, "node_power"].quantile(
                                   [0.05, 0.25, 0.5, 0.75, 0.95]).round(0).to_dict()}

    # Test-set predictions (private: keyed by job id) so later analysis needs no retraining
    out = pd.DataFrame({"job_id": test.job_id.to_numpy(), "y_log_runtime": y,
                        **{f"rt_{k}": v for k, v in preds.items()}})
    out.to_parquet(a.out / "runtime_preds.parquet", index=False)
    pd.DataFrame({"job_id": ptest.job_id.to_numpy(), "y_node_power": yp, **{f"pw_{k}": v for k, v in ppred.items()}}
                 ).to_parquet(a.out / "power_preds.parquet", index=False)
    (a.out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(report(results))
    (a.out / "report.md").write_text(report(results))


def report(r):
    lines = [f"Training weights: {r.get('train_weight', 'none')}. Split: {r['split']}", ""]
    for key, title in (("runtime", "per job"), ("runtime_nh", "weighted by requested node-hours")):
        lines += [f"## Runtime, {title} (test set; R² on log runtime)", "",
                  "| predictor | subset | n | R² log | median abs err (min) | median rel err | within ±25% | under-predicted >20% |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, by in r[key].items():
            for s, m in by.items():
                lines.append(f"| {name} | {s} | {m['n']:,} | {m['r2_log']:.3f} | {m['medae_min']:.1f} | "
                             f"{m['median_rel_err']:.2f} | {m['within_25pct']:.1%} | {m['under_pred']:.1%} |")
        lines.append("")
    for key, title in (("power", "per job"), ("power_nh", "weighted by requested node-hours")):
        lines += [f"## Mean node power, {title} (test set)", "",
                  "| predictor | subset | n | R² | MAE (W/node) | median rel err |", "|---|---|---|---|---|---|"]
        for name, by in r[key].items():
            for s, m in by.items():
                lines.append(f"| {name} | {s} | {m['n']:,} | {m['r2']:.3f} | {m['mae_w']:.0f} | {m['median_rel_err']:.2f} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
