#!/bin/bash
# Waits until one of the given PIDs ends (polls every 20 s, at most MAX_S s, default 7000), then prints which one
# ended, the pause log lines and the GPU memory. Usage: bash wait_any_pid_end.sh <pid> [pid ...]
MAX_S="${MAX_S:-7000}"; END=$((SECONDS + MAX_S))
while [ "$SECONDS" -lt "$END" ]; do
  for p in "$@"; do
    if ! kill -0 "$p" 2>/dev/null; then
      echo "$(date -u +%T) PID $p ended"
      tail -n 3 ${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/g166b_*_rollfix/logs/pause.log 2>/dev/null | cut -c1-220
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
      exit 0
    fi
  done
  sleep 20
done
echo "$(date -u +%T) none ended in $MAX_S s"; exit 3
