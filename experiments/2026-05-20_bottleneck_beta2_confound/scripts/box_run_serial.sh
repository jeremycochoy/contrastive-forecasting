#!/bin/bash
# #309 box-side serial driver — runs alpha then beta then gamma on one
# vast box. Idempotent: each arm calls box_run.sh which skips on FINAL.
# Priority order (per issue): alpha > beta > gamma.
#
# Usage (ON BOX):  bash box_run_serial.sh
set -uo pipefail
APP=/workspace/app
cd "$APP"
SC="$APP/experiments/2026-05-20_bottleneck_beta2_confound/scripts/box_run.sh"
LG="$APP/results/serial.log"
mkdir -p "$APP/results"
echo "=== SERIAL START $(date) ===" | tee -a "$LG"
for ARM in alpha beta gamma; do
  echo "[$(date '+%m-%d %H:%M:%S')] === serial: $ARM ===" | tee -a "$LG"
  bash "$SC" "$ARM" || echo "[$(date '+%m-%d %H:%M:%S')] arm $ARM rc=$?" | tee -a "$LG"
done
echo "=== SERIAL DONE $(date) ===" | tee -a "$LG"
