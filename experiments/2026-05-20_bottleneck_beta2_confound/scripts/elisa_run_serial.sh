#!/bin/bash
# #309 follow-up serial driver — α→β→γ at a given τ on elisa (DDP both
# GPUs). Idempotent (each arm skips if its FINAL exists). A diverged
# no-bneck arm can be SIGTERM'd from outside; elisa_run.sh then copies
# best_loss→FINAL and this driver advances to the next arm.
#
# Usage:  elisa_run_serial.sh <tau> <runtag> [gpus]
set -uo pipefail
TAU="${1:?tau}"; RUNTAG="${2:?runtag}"; GPUS="${3:-0,1}"
SC="$(dirname "$0")/elisa_run.sh"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound
LG="$MAIN/results/serial_${RUNTAG}.log"
mkdir -p "$MAIN/results"
echo "=== SERIAL ${RUNTAG} (τ=$TAU) START $(date) ===" | tee -a "$LG"
for ARM in alpha beta gamma; do
  echo "[$(date '+%m-%d %H:%M:%S')] === serial: $ARM (τ=$TAU) ===" | tee -a "$LG"
  bash "$SC" "$ARM" "$TAU" "$RUNTAG" "$GPUS" || echo "[$(date '+%m-%d %H:%M:%S')] arm $ARM rc=$?" | tee -a "$LG"
done
echo "=== SERIAL ${RUNTAG} DONE $(date) ===" | tee -a "$LG"
