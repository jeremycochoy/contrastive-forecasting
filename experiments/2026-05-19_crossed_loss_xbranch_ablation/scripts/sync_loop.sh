#!/bin/bash
# Persistent sync loop (run detached on elisa for the FULL run duration —
# CLAUDE.md REMOTE_LAUNCH_CHECKLIST). Ticks every 15 min over all arms
# that have a state/<arm>.env. Writes sync.log; pidfile for liveness.
#   nohup setsid bash sync_loop.sh <arm...> > sync.log 2>&1 &
set -uo pipefail
EXP=/home/jupyter/cf-wt-crossed-loss/experiments/2026-05-19_crossed_loss_xbranch_ablation
INTERVAL=900
ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(hhff fhhhff hhxbf)
echo $$ > "$EXP/scripts/state/sync_loop.pid"
echo "sync_loop start $(date) arms=${ARMS[*]} interval=${INTERVAL}s"
while true; do
  for a in "${ARMS[@]}"; do
    [ -f "$EXP/scripts/state/$a.env" ] || continue
    bash "$EXP/scripts/sync_arm.sh" "$a" || echo "[$a] tick error (continuing)"
  done
  echo "--- sleeping ${INTERVAL}s $(date '+%H:%M:%S') ---"
  sleep "$INTERVAL"
done
