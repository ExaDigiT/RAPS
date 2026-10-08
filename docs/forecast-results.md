# Per-job runtime and power prediction on Frontier: first results (2026-10-08)

Data: `sacct -a` for 2024-01-01 to 2026-10-08, giving 3.74 M jobs that ran, from 2,261 users. It is
kept privately on OLCF systems; this file holds aggregate numbers only. Code: `raps/forecast/`.
Batch jobs used about 0.45 node-hours (stf053, extended partition).

**Split:**
- train: submitted before 2026-01-01 (2.13 M jobs)
- validation: 2026-01-01 to 2026-04-01 (251 k)
- test: 2026-04-01 to 2026-09-24 (1.13 M)
- held-out users: a stable 10% of users, dropped from train and validation (84 k test jobs)
- first runs: test jobs with no earlier finished run of the same user and job name (175 k)

## Read this first: per-job scores mostly measure a few accounts

One user submitted 31.6% of all test jobs, and the top 10 submitted 52.5% of them but used 4.3% of
node-hours. Those jobs are mostly 1-node runs of under 2 minutes. Per-job averages are dominated by
them, and they hardly affect when other jobs start. The headline tables below therefore weight
each job by its **requested node-hours** (nodes × time limit). That weight is known at submission
and does not depend on the outcome.

(Weighting by *actual* node-hours, which an earlier run did, favours jobs that happened to run long.
It made "predict the time limit" look best, at R² 0.76. That was an artifact.)

## Runtime, weighted by requested node-hours (test set)

| predictor | R² (log) | median abs err | median rel err | within ±25% | under-predicted by >20% |
|---|---|---|---|---|---|
| time limit (Slurm's assumption) | −0.45 | 80 min | 1.14 | 31.5% | 0% |
| history rule | 0.24 | **39 min** | **0.41** | **39.8%** | 31.8% |
| LightGBM, requested resources only | 0.11 | 70 min | 0.85 | 6.6% | 65.1% |
| LightGBM, + MiniLM identity embedding | 0.10 | 62 min | 0.89 | 5.7% | 68.9% |
| LightGBM, + history | **0.41** | 55 min | 0.68 | 16.3% | 56.2% |
| LightGBM, + history + embedding | 0.40 | 54 min | 0.70 | 14.2% | 59.0% |

The history rule is the mean runtime ÷ time-limit ratio of the last 5 finished runs of the same
user and job name, falling back to the same user, then the same account. It is multiplied by the
job's time limit.

**By job size** (median relative error, weighted by requested node-hours):

| nodes | jobs | share of requested node-hours | ran to ≥95% of limit | time limit | history rule | LightGBM + history |
|---|---|---|---|---|---|---|
| 1–15 | 1,006 k | 3.4% | 26% | 1.87 | **0.24** | 0.50 |
| 16–127 | 79 k | 9.2% | 27% | 0.58 | **0.39** | 0.65 |
| 128–1023 | 38 k | 42.4% | 26% | 0.87 | **0.27** | 0.69 |
| ≥1024 | 7 k | 45.0% | 22% | 1.52 | **0.55** | 0.68 |

**Held-out users and first runs** (R², weighted):

| predictor | held-out users | first runs |
|---|---|---|
| history rule | 0.02 | 0.01 |
| LightGBM + history | 0.22 | 0.20 |
| LightGBM + history + embedding | 0.22 | 0.20 |

Training weights barely matter: unweighted, per-user (1/√ job count) and requested-node-hour
weighting all land within ±0.04 R² of each other. Per-job (unweighted) results are in
`results_*/report.md` in the data directory. There the history rule has a median error of 1.1 min,
and LightGBM + history reaches R² 0.66.

## Mean power per node (energy ÷ runtime ÷ nodes; energy recorded from mid-June 2024)

| predictor | R² (weighted) | MAE | median rel err |
|---|---|---|---|
| history rule | 0.38 | 239 W | **0.10** |
| LightGBM, requested resources only | 0.05 | 421 W | 0.42 |
| LightGBM, + history | 0.42 | 271 W | 0.20 |
| LightGBM, + history + embedding | **0.43** | 261 W | 0.18 |

Readings outside 50–5,000 W per node (energy-counter glitches up to ~10⁹ W) are treated as missing.

## Findings

1. **Slurm's assumption is far off.** Only 22–27% of requested node-hours run to the limit. The
   history rule halves the median error (80 → 39 min) and is the best predictor of typical error for
   every job size.
2. **What a job requests says little about its runtime or power** (R² ≈ 0.1 and ≈ 0.05). This
   agrees with the SC26 paper's finding for static scheduler metadata.
3. **The identity embedding adds nothing on Frontier in this setup.** On its own it scores like
   requested-only, and on top of history it changes R² by −0.01 to +0.01. Caveat: the trees see 32
   PCA components (68% of the variance), not a neural head on the full 384 dimensions as in the
   paper. The paper's 0.960 is also a window-mean target on Fugaku, not per-job. A fair test is
   still to do (next steps).
4. **Unseen users stay hard** (weighted R² ≤ 0.22 for every model). Nearly all of the predictability
   comes from a user's own earlier runs. This supports the paper's footnote that identity-conditioned
   scores partly reflect recalling known users.
5. **LightGBM trades typical error for R².** Squared error on log runtime pulls predictions toward
   the middle, so it under-predicts big jobs (56% of requested node-hours by more than 20%). For
   queue forecasting the history rule is the better default today.

## Next steps

- A fair embedding test: an MLP on the full 384-d MiniLM vector plus history, and an ExaFM per-job
  head, on these same splits.
- Better objectives for the trees: quantile or asymmetric loss, or a blend with the history rule.
- Put the history rule in the live console now, as a "forecast runtime" next to Slurm's estimate,
  and feed it to a RAPS forecast of start times. Back-test against actual starts using saved queue
  snapshots.
- Check `sacct` energy against measured power in the public Job-Centric dataset (distributions only).
