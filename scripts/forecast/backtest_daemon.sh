#!/bin/bash
# Forecast back-test daemon: record a snapshot every 10 min (relaunched if it ever exits) and
# score the snapshots hourly into $D/report.md. Run it in tmux on a login node:
#   tmux new-session -d -s raps-backtest "scripts/forecast/backtest_daemon.sh"
D=${1:-$PROJWORK/stf053/exadigit/forecast-data/backtest}
PY=${PYTHON:-/autofs/nccs-svm1_home1/whbrewer/venv/exadigit/bin/python}
PROBE_ACCOUNT=${PROBE_ACCOUNT:-stf053}
umask 077
mkdir -p "$D"
cd "$(dirname "$0")/../.." || exit 1

(while true; do
    sleep 3600
    "$PY" -m raps.live.backtest evaluate --dir "$D" > "$D/evaluate.log" 2>&1
done) &

while true; do
    "$PY" -m raps.live.backtest record --dir "$D" --every 10 --probe-account "$PROBE_ACCOUNT" >> "$D/record.log" 2>&1
    echo "$(date '+%F %T') recorder exited ($?); relaunching in 60 s" >> "$D/record.log"
    sleep 60
done
