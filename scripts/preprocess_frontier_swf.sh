#!/usr/bin/env bash
#
# Preprocess the public Frontier SWF workload traces into a RAPS-native
# raps_jobs.npz snapshot, using REDI's SWF loader + RAPS job-extraction
# pipeline (REDI: https://github.com/ORNL/redi, expected at ~/drai/redi).
#
# Source data (Standard Workload Format, Parallel Workloads Archive style):
#   /opt/data/hpc/frontier/frontier-2024.swf
#   /opt/data/hpc/frontier/frontier-2025.swf
#
# This is scheduler-scalar data only (submit/wait/run times, node counts) --
# there are no per-job power/CPU/GPU utilization traces in SWF, so the
# resulting replay drives job scheduling/queuing but not telemetry-based
# power modeling.
#
# Usage:
#   scripts/preprocess_frontier_swf.sh [output_dir]
#
# Produces <output_dir>/raps_jobs.npz (default: data/frontier_swf), replay with:
#   raps run --replay <output_dir>/raps_jobs.npz --system frontier
# or simply:
#   raps run experiments/frontier-swf.yaml

set -euo pipefail

REDI_DIR="${REDI_DIR:-$HOME/drai/redi}"
OUT_DIR="${1:-data/frontier_swf}"

if [ ! -d "$REDI_DIR" ]; then
    echo "REDI not found at $REDI_DIR (clone it, or set REDI_DIR to override)" >&2
    exit 1
fi

PYTHONPATH="$REDI_DIR:${PYTHONPATH:-}" python -m redi.cli run \
    -c "$REDI_DIR/examples/hpc/frontier_swf.yaml" \
    -o "$OUT_DIR" \
    --style simple

echo
echo "Done. Replay with:"
echo "  raps run --replay $OUT_DIR/raps_jobs.npz --system frontier"
echo "or: raps run experiments/frontier-swf.yaml"
