#!/bin/bash
# Samples system memory and a process tree's total RSS every INTERVAL seconds.
# Usage: monitor_memory.sh <root_pid> <log_file> [interval_seconds]
ROOT_PID=$1
LOG_FILE=$2
INTERVAL=${3:-15}

echo "timestamp,mem_used_gb,mem_avail_gb,swap_used_gb,tree_rss_gb,num_procs,load1" > "$LOG_FILE"

while kill -0 "$ROOT_PID" 2>/dev/null; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    read -r MEM_USED MEM_AVAIL <<< "$(free -m | awk '/^Mem:/{printf "%.2f %.2f", $3/1024, $7/1024}')"
    SWAP_USED=$(free -m | awk '/^Swap:/{printf "%.2f", $3/1024}')
    LOAD1=$(cut -d' ' -f1 /proc/loadavg)

    # Sum RSS (KB) across the whole process tree rooted at ROOT_PID
    PIDS=$(pstree -p "$ROOT_PID" 2>/dev/null | grep -oP '\(\K[0-9]+(?=\))')
    if [ -z "$PIDS" ]; then
        PIDS=$ROOT_PID
    fi
    TREE_RSS_KB=$(ps -o rss= -p $(echo $PIDS | tr ' ' ',') 2>/dev/null | awk '{sum+=$1} END{print sum+0}')
    TREE_RSS_GB=$(echo "$TREE_RSS_KB" | awk '{printf "%.2f", $1/1024/1024}')
    NUM_PROCS=$(echo "$PIDS" | wc -w)

    echo "$TS,$MEM_USED,$MEM_AVAIL,$SWAP_USED,$TREE_RSS_GB,$NUM_PROCS,$LOAD1" >> "$LOG_FILE"
    sleep "$INTERVAL"
done
echo "$(date '+%Y-%m-%d %H:%M:%S'): root pid $ROOT_PID no longer alive, stopping monitor" >> "$LOG_FILE"
