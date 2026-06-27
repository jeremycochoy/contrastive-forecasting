#!/bin/bash
# #363 — arm-6 supervisor: wait for queue_arm6.sh to exit, then run finish_arm6.sh.
# Designed to be nohup'd in parallel with the queue.
set -uo pipefail
WT=/tmp/cf-revert-363
OUT="$WT/experiments/2026-06-24_sigreg_lambda_sweep"
SLOG="$OUT/results/supervisor_arm6.log"
mkdir -p "$OUT/results"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [supervisor-arm6] $*" | tee -a "$SLOG" >&2; }

QPID="${1:?queue pid}"
log "supervisor start — watching queue PID=$QPID"

# Wait for the queue process to exit (poll every 60s)
while kill -0 "$QPID" 2>/dev/null; do
  sleep 60
done
log "queue PID $QPID has exited"

# Inspect queue log to determine success
QLOG="$OUT/results/queue_arm6.log"
if ! grep -q "queue done" "$QLOG" 2>/dev/null; then
  log "queue_arm6.log did not record 'queue done' — refusing to run finisher"
  log "last 20 lines of queue log:"
  tail -20 "$QLOG" 2>/dev/null | tee -a "$SLOG"
  exit 1
fi

log "queue succeeded — running finish_arm6.sh"
bash "$OUT/scripts/finish_arm6.sh"
rc=$?
log "finish_arm6.sh rc=$rc"
log "supervisor done"
