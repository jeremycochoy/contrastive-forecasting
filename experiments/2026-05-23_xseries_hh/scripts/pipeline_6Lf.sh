#!/bin/bash
# #318 follow-up — full 6L-forecaster pipeline (user request). Trains + evals
# both arms with a 6-layer forecaster, ALL-TIME FIRST (most negative terms),
# then same-step. Each: train backbone, then 2L + 6L q-heads × {triage, full}.
# Idempotent (skips finished cells). Run on one GPU.
#
#   pipeline_6Lf.sh <gpu> [chunk]
set -uo pipefail
GPU="${1:?gpu}"; CHUNK="${2:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [6Lf-pipeline g$GPU] $*"; }

# --- arm with the MOST negatives first: all-time ---
log "=== ALL-TIME 6L-forecaster (most negatives) ==="
bash "$HERE/train_backbone_6Lf.sh" alltime "$GPU" "$CHUNK"
AB="$OUT/runs/bb_xshh_allt_6Lf_50k_FINAL.pth"
if [ -f "$AB" ]; then
  bash "$HERE/downstream_6Lf.sh" "$AB" 2 xshh_allt_6Lf_50k "$GPU" "triage full"
  bash "$HERE/downstream_6Lf.sh" "$AB" 6 xshh_allt_6Lf_50k "$GPU" "triage full"
else log "all-time 6Lf backbone missing — skipping its eval"; fi

# --- then same-step ---
log "=== SAME-STEP 6L-forecaster ==="
bash "$HERE/train_backbone_6Lf.sh" samestep "$GPU" "$CHUNK"
SB="$OUT/runs/bb_xshh_6Lf_50k_FINAL.pth"
if [ -f "$SB" ]; then
  bash "$HERE/downstream_6Lf.sh" "$SB" 2 xshh_6Lf_50k "$GPU" "triage full"
  bash "$HERE/downstream_6Lf.sh" "$SB" 6 xshh_6Lf_50k "$GPU" "triage full"
else log "same-step 6Lf backbone missing — skipping its eval"; fi

log "=== PIPELINE_6Lf_DONE ==="
