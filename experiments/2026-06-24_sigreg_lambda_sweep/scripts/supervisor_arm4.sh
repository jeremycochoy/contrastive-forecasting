#!/bin/bash
# #363 — arm-4 supervisor: wait for queue_arm4.sh to exit, then run finish_arm4.sh.
# Designed to be nohup'd in parallel with the queue.
set -uo pipefail
WT=/tmp/cf-revert-363
OUT="$WT/experiments/2026-06-24_sigreg_lambda_sweep"
SLOG="$OUT/results/supervisor_arm4.log"
mkdir -p "$OUT/results"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [supervisor-arm4] $*" | tee -a "$SLOG" >&2; }

QPID="${1:?queue pid}"
log "supervisor start — watching queue PID=$QPID"

while kill -0 "$QPID" 2>/dev/null; do
  sleep 60
done
log "queue PID $QPID has exited"

QLOG="$OUT/results/queue_arm4.log"
if ! grep -q "queue done" "$QLOG" 2>/dev/null; then
  log "queue_arm4.log did not record 'queue done' — refusing to run finisher"
  log "last 20 lines of queue log:"
  tail -20 "$QLOG" 2>/dev/null | tee -a "$SLOG"
  exit 1
fi

log "queue succeeded — running finish_arm4.sh"
bash "$OUT/scripts/finish_arm4.sh"
rc=$?
log "finish_arm4.sh rc=$rc"
log "supervisor done"
