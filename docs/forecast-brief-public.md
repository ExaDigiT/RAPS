# Brief: per-job runtime and power on the public Frontier Job-Centric dataset

For a Claude Code session on PC135767, where the public dataset lives. This is the
public-data track of the RAPS job-forecasting work. The internal track (real identities from
`sacct -a`) runs only on Frontier and must not be copied here.

## Data

`/store/datasets/hpc/frontier/telemetry/frontier-completed-job-info.parquet` is the Frontier
Job-Centric Telemetry Dataset (DOI 10.13139/OLCF/3013979):

- 152,400 jobs ending between 2024-01-17 and 2026-01-01, a 6.8% sample from 135 days
- job names, scripts and real identities were removed
- `user_id` is a stable pseudonym for each (user, project) pair; `project_id` and
  `project_prefix` are stable pseudonyms too
- columns used here: `submit_time`, `start_time`, `end_time`, `wall_time` (time limit), `run_time`,
  `node_count`, `queue`, `job_mode`, `gpu_enabled`, `mean_power`, `peak_power`, `user_id`,
  `project_id`, `project_prefix`, `completion_state`

Per-job power and temperature time series are under `job-telemetry/<date>/<job_idx>/`. RAPS reads
them with `raps/dataloaders/frontier_jobcentric.py`.

## Question

How much does identity help per-job prediction when only pseudonyms are available, compared with
having no identity at all? This gives a public, reproducible counterpart to the SC26 paper's
F-Data result (R² 0.960 with a pre-anonymization Sentence-BERT embedding versus −1.669 without).

## What to build (mirror `raps/forecast/` on the `forecast` branch of code.ornl.gov/sc/raps)

1. **Targets:** `log1p(run_time)`; `mean_power` per node (check the units first: total or per node,
   W or kW; Frontier nodes typically draw 500–2,500 W).
2. **Features at submit time:** log time limit, log nodes, queue, job_mode, gpu_enabled, submit
   hour and day of week. Also history for the same `user_id`, `project_id` and `project_prefix`,
   using only jobs that **ended before this job was submitted** (copy `_history()` from
   `raps/forecast/dataset.py`; its tests are in `tests/test_forecast.py`). Expect sparse history,
   because the sample is 6.8% of jobs across scattered days.
3. **Predictors, from simplest to richest** (same as the Frontier track):
   - time limit
   - history rule
   - LightGBM on what was requested
   - LightGBM with history
   - LightGBM with a learned pseudonym embedding (LightGBM categorical `user_id`/`project_id`, or a
     small MLP with an embedding table)
   - optionally, an ExaFM per-job head
4. **Splits:**
   - chronological by submit time
   - a held-out set of 10% of `user_id`s that never appear in training (use the same hash as
     `holdout_users()` in `raps/forecast/evaluate.py`)
   - report all test jobs, held-out users, and first runs
5. **Metrics:** R² on log runtime, median absolute error in minutes, median relative error, share
   within ±25%, share under-predicted by more than 20%. For power: R², MAE, median relative error.

## Notes

- One person on several projects appears as several `user_id`s, so held-out "users" are really
  held-out (user, project) pairs. Say so when reporting.
- Do not try to re-link records to real Slurm jobs, users or job names. The timestamps plus
  `host_list` would probably make that possible, which is a privacy finding to report to the
  dataset authors, not something to use.
- Keep results and code in a branch of the same repo so the two tracks can be compared.
