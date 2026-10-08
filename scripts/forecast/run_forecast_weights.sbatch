#!/bin/bash
#SBATCH -A stf053
#SBATCH -J raps-forecast-weights
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p extended
#SBATCH -o %x-%j.out
# Rerun only the predictor comparison (jobs.parquet and embeddings.parquet already built) under
# three training weightings. Results go to $D/results_<weight>/.
#   sbatch -o $D/logs/%x-%j.out scripts/forecast/run_forecast_weights.sbatch
set -euo pipefail
D=${FORECAST_DATA:-$PROJWORK/stf053/exadigit/forecast-data}
PY=${PY:-/autofs/nccs-svm1_home1/whbrewer/venv/exadigit/bin/python}
export OMP_NUM_THREADS=56
cd ${RAPS_DIR:-$HOME/exadigit/raps}
umask 077
for w in none user nodehours; do
  date; echo "== train-weight $w"
  $PY -m raps.forecast.evaluate --jobs $D/jobs.parquet --emb $D/embeddings.parquet --out $D/results_$w \
      --train-weight $w --threads 56
done
date
