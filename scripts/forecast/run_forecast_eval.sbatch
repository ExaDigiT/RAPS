#!/bin/bash
#SBATCH -A stf053
#SBATCH -J raps-forecast-eval
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH -o %x-%j.out
# Build the per-job table from the sacct pulls, embed identities with MiniLM, and run the
# predictor comparison. Data stays in the private forecast-data directory.
#   sbatch scripts/forecast/run_forecast_eval.sbatch
set -euo pipefail
D=${FORECAST_DATA:-$PROJWORK/stf053/exadigit/forecast-data}
PY=${PY:-/autofs/nccs-svm1_home1/whbrewer/venv/exadigit/bin/python}
export HF_HOME=$D/hf-cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=56 TOKENIZERS_PARALLELISM=false
cd ${RAPS_DIR:-$HOME/exadigit/raps}
umask 077
date
$PY -m raps.forecast.dataset --sacct $D/sacct --out $D/jobs.parquet
date
$PY -m raps.forecast.embed --jobs $D/jobs.parquet --out $D/embeddings.parquet
date
$PY -m raps.forecast.evaluate --jobs $D/jobs.parquet --emb $D/embeddings.parquet --out $D/results --threads 56
date
