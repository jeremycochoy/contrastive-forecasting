#!/bin/bash
# #313 driver — full chain on one elisa GPU: train → downstream → plots.
# Idempotent (each stage skips if its FINAL/summary already exists).
#
#   driver.sh <gpu_id>     gpu_id = single GPU, e.g. 1
set -uo pipefail
GPU="${1:?gpu_id (single, e.g. 1)}"
SC="$(cd "$(dirname "$0")" && pwd)"
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B
RES="$MAIN/results"; mkdir -p "$RES"
LG="$RES/driver.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [driver g$GPU] $*" | tee -a "$LG"; }

log "=== CHAIN START on GPU $GPU ==="
bash "$SC/elisa_run.sh" "$GPU" 2>&1 | tee -a "$LG"
[ -f "$MAIN/runs/bb_alignfloor_50k_FINAL.pth" ] || { log "ABORT: training produced no FINAL backbone"; exit 1; }
bash "$SC/downstream.sh" "$GPU" 2>&1 | tee -a "$LG"
log "--- plots ---"
python3 "$SC/plot_loss.py"    2>&1 | tee -a "$LG" || true
python3 "$SC/plot_radar.py"   2>&1 | tee -a "$LG" || true
python3 "$SC/plot_summary.py" 2>&1 | tee -a "$LG" || true
echo "ALIGNFLOOR_CHAIN_DONE $(date -u +%FT%TZ)" > "$RES/.chain_done"
log "=== CHAIN DONE ==="
