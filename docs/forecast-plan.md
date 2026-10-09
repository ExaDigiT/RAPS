# Forecasting plan: scoreboard daemon, fair ML test, twin improvements (2026-10-08)

## Where things stand

- `raps live` (branch `live`) shows three kinds of forecast:
  - per-job runtime and W/node from the **history rule**: the mean of the last 5 finished runs
    of the same job, else the same user, else the same project
  - **RAPS twin** start times: the live queue run through RAPS's scheduler
  - **system power** over the next 24 h
- Offline comparison (`docs/forecast-results.md`, 3.7 M jobs):
  - the history rule has the best typical error
  - LightGBM has a higher R² but under-predicts big jobs
  - the identity embedding, as tested (32 PCA components in trees), adds nothing; this was
    **not** a fair test of ExaFM
- `python -m raps.live.backtest record` runs in tmux session `raps-backtest` on login05. It
  snapshots the predictions every 10 min into `$PROJWORK/stf053/exadigit/forecast-data/backtest/`
  (no user, project or job names; about 45 MB/day). `evaluate` scores them once.
  Snapshots before 2026-10-08 23:17 used older backfill rules.
- Upstream fixes are on `fix-backfill-shadow` (private remote only, merged into `live`):
  - backfill shadow-time ordering (`default.py`, `experimental.py`)
  - EASY backfill reordering the queue in place
  - the extra-nodes rule
  - the always-true `bfpolicy` check (all three schedulers)
  - the missing `policy` argument in `experimental.py`

## A. Forecast scoreboard daemon (next)

Turn the recorder into a long-running service that also *scores itself*, so forecast quality is
tracked over days and weeks, and new models can be scored live before they reach the console.

**One process, three loops** (`python -m raps.live.scoreboard --dir $D`):

1. **record** (every 10 min; exists): pending and running jobs, the twin, and system power.
2. **resolve** (hourly): look up actual start, end, energy and state with `sacct` for jobs in
   recent snapshots. Append scored rows (prediction, actual, error, lead time, size, basis) to
   daily Parquet files.
3. **summarize** (hourly): roll scored rows into a metrics table (`metrics.parquet`: one row per
   metric × predictor × group × window) and write `report.md`.

**Statistics to track** (each for the last 24 h, the last 7 days and all time):

| target | predictors | broken down by | metrics |
|---|---|---|---|
| job runtime | history rule, time limit (+ shadow models, below) | basis (job/user/project/none), job size, first run vs repeat | median abs error, median rel error, within ±25%, under-predicted >20%, R² (log), all weighted by requested node-hours |
| "past forecast" rate | history rule | basis, size | share of running jobs that outlive their forecast |
| start time | twin, twin at limits, Slurm | lead time (<1 h, 1–6 h, 6–24 h, >24 h), size, partition | median abs error, bias, within ±25% of the wait, rank correlation of start order |
| job power | history rule (W/node) vs actual energy ÷ runtime | basis, size | MAE, median rel error |
| system power | twin vs persistence | lead (+0.5 … +24 h) | MAE, bias, peak timing error, peak size error |
| nodes freed within 1 h | forecast vs time limit | — | abs error against actual nodes freed |
| coverage | — | — | share of jobs with a forecast at each basis; share never started within 14 d |

**Shadow mode for new models.** A small `Predictor` interface (`predict(job) -> Prediction`) and
a registry. The daemon runs every registered predictor on each snapshot and stores all their
predictions, so a new model (an MLP, an ExaFM head) is scored on live traffic without being
switched into the console. The console keeps the default (the history rule, or whatever wins).

**Console view.** A "Forecasts" tab (or Overview panel) that reads `metrics.parquet`. It shows
the scoreboard (last 24 h and 7 days), and in-console sparklines of each metric over time.

**Running it in the background.** In order of preference:

1. **tmux on a login node** with a restart wrapper (`scripts/forecast/scoreboard.sh`: a loop
   that relaunches on exit and logs to `$D/scoreboard.log`). Cheap, as now. A login node reboot
   kills it, so pair it with:
2. **`scrontab`** (available on Frontier) for an hourly check that the daemon is alive (and to
   restart it if not). Note that scrontab jobs run on compute nodes and are charged to the
   allocation. A 1-node `-t 5:00` check every hour is about 2 node-hours a week. As an
   alternative, put only the hourly resolve and summarize loops in scrontab and keep recording
   on the login node.
3. Ask OLCF about a sanctioned place for long-running user services, if this grows.

**Storage and privacy.**
- gzip snapshot days older than 2 days, and keep scored Parquet indefinitely (a few MB/day).
- Mode 700 and `umask 077` as now. No names in any row.
- Shadow-model predictions are numbers only.

**Effort:** about a day.
- The resolve and summarize loops, plus tests: half a day.
- The predictor registry and the console tab: half a day.

## B. Fair ML test (after A, so winners are scored live)

Question: does a learned model beat the history rule where it matters, and above all on
**first runs and new users**, where the history rule has R² ≈ 0?

1. **Quick baseline:** an MLP on the full 384-d MiniLM identity embedding + request + history
   features. Same splits and metrics as `raps/forecast/evaluate.py` (requested-node-hour
   weights, held-out users, first runs). Loss options: log-space MSE, plus a quantile/asymmetric
   variant to stop under-prediction of big jobs.
2. **ExaFM per-job head:** add a per-job runtime/power head to ExaFM (its current `job_runtime`
   head is a window mean), fine-tune on the same training split, and evaluate identically.
3. **Hybrid:** the history rule when the job has its own history (● green), the learned model
   otherwise. This is likely the best practical combination.
4. **Decision rule:** a model goes into the console if, on the test split and then in
   shadow mode for about a week, it beats the history rule on:
   - weighted median relative error
   - under-predicted share (it must not be worse)

   It can win overall or only for the fallback cases.
5. **Cost:** Frontier GPU jobs under stf053, within the 10 node-hour cap (about 0.45 used).
   The model file is trained on identity embeddings and stays in the private data directory.

## C. Twin improvements (driven by the scoreboard)

Each change should be judged by start-time and power scores before and after.

- **Arrivals:** inject new submissions, by resampling recent submission patterns by hour of
  day. Without them, the twin's dips after big jobs and its late-horizon drain are artifacts.
- **Priority aging:** Slurm's age factor, so long-waiting jobs move up.
- **Jobs that become eligible later:** dependencies (start after the parent's forecast end)
  and holds.
- Per-user and per-project limits (QOS) and future reservations.
- The batch-spi node subset; jobs listed under several partitions (currently placed in the
  first one listed).
- Idle power: the RAPS model says 652 W/node and Frontier measures about 476 W. Calibrate the
  config or note the gap.

## D. Upstream

- When ready, open a merge request from `fix-backfill-shadow` into exadigit/raps `main`.
- Rerun any published or ongoing RAPS studies that used `--backfill firstfit` or `--backfill
  easy` since 2025-08-22 (commit `f2b77f8`), since their queue-wait results will shift.

## Decisions for Wes

1. Scoreboard: tmux only, or tmux plus an hourly scrontab check (charged to stf053)?
2. ML test order: the MLP baseline first, or straight to an ExaFM per-job head?
3. Console placement of the scoreboard: a new tab, or a panel on Overview?
4. Timing of the upstream merge request.
