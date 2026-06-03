#!/bin/bash
# Quick status check for the running experiment.
set -uo pipefail
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== process ==="
PID=$(cat "$EXP_DIR/logs/corrV1.pid" 2>/dev/null || echo "")
if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
    ps -p "$PID" -o pid,stat,etime,%cpu,%mem,cmd
else
    echo "No PID or process not running."
    pgrep -af train_contrastive_corr.py || echo "No train_contrastive_corr.py process found."
fi

echo ""
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv

echo ""
echo "=== latest log lines ==="
tail -n 20 "$EXP_DIR/logs/corrV1.log" 2>/dev/null || echo "(no log)"

echo ""
echo "=== checkpoints ==="
ls -lah "$EXP_DIR/checkpoints/" 2>/dev/null | tail -n 20 || echo "(no checkpoints dir)"

echo ""
echo "=== recent figures ==="
ls -lah "$EXP_DIR/plots/" 2>/dev/null | tail -n 20 || echo "(no figures dir)"
