#!/bin/bash
# #401 — one liveness probe, then wait.
#
# The study has about 17 h of phase 1 left and phase 2 after it. A watcher
# that only fires on completion misses a stall: a process that hangs
# without exiting never sends the event. So this script probes, then blocks
# for at most one hour, and returns early when a milestone lands.
#
# It probes, it does not repair. The caller reads the probe and decides.
#
# Exit codes:
#   0  the hour ended, the study runs                (heartbeat)
#   10 a new GIFT-Eval score landed                  (progress)
#   11 the launcher is gone                          (stall or end)
#   12 the training step counter did not advance     (stall)
#
# Usage:  bash scripts/heartbeat.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
LAUNCHER_PID="${LAUNCHER_PID:-542935}"
MAX="${MAX:-3300}"          # 55 min, so the caller wakes inside the hour
TICK="${TICK:-120}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 hb] $*" \
  | tee -a "$RES/heartbeat.log"; }

# The live training step, from whichever arm log moved last.
cur_step(){
  local f
  f="$(ls -t "$RES"/run_*_cf373k*.log 2>/dev/null | grep -v smoke | head -1)"
  [ -n "$f" ] || { echo 0; return; }
  grep -oE '^\[ *[0-9]+\]' "$f" | tail -1 | tr -dc '0-9'
}

n_scores(){ ls "$RES"/score_*.txt 2>/dev/null | wc -l; }

s0="$(cur_step)"; c0="$(n_scores)"
log "probe: launcher=$( ps -p "$LAUNCHER_PID" >/dev/null && echo alive || echo GONE ) step=$s0 scores=$c0 gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"

waited=0
while [ "$waited" -lt "$MAX" ]; do
  sleep "$TICK"; waited=$(( waited + TICK ))
  ps -p "$LAUNCHER_PID" >/dev/null 2>&1 || { log "launcher $LAUNCHER_PID GONE"; exit 11; }
  c="$(n_scores)"
  [ "$c" -gt "$c0" ] && { log "new score, $c0 -> $c"; exit 10; }
done

s1="$(cur_step)"
log "hour done: step $s0 -> $s1, scores $c0"
# A backbone leg runs at about 1.6 steps per second, so an hour with no new
# step means the trainer stopped moving even though the process is alive.
if [ "${s1:-0}" -le "${s0:-0}" ] && [ "$(n_scores)" -eq "$c0" ]; then
  log "STALL: the step counter did not advance in $MAX s"
  exit 12
fi
exit 0
