#!/bin/bash
# #374 arm 3 — 15-min sync loop from the arm-3 vast.ai instance to elisa.
# Pulls losses.csv, attn_amplitude.csv, run.log, best/last backbones,
# best/last qhead checkpoints, and per-cell all_results.csv + summary.txt.
# Exits when all four downstream summary.txt files are present locally.
set -uo pipefail
HOST=ssh8.vast.ai
PORT=34324
REMOTE=/workspace/cf-374/experiments/2026-07-10_split_pred_rep
LOCAL=/tmp/contrastive-forecasting-374/experiments/2026-07-10_split_pred_rep
SAFE_PULL=/tmp/contrastive-forecasting-374/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh
INTERVAL=900
NAME=bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090
TAG=split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090
CELLS=("${TAG}_2L" "${TAG}_last_2L" "${TAG}_6L" "${TAG}_last_6L")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sync-arm3] $*"; }
pull(){ bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$2" "$3" || echo "  (skip: $1)"; }

all_done(){
  for c in "${CELLS[@]}"; do
    [ -f "$LOCAL/results/gift_eval_full_$c/summary.txt" ] || return 1
  done
  return 0
}

mkdir -p "$LOCAL/runs" "$LOCAL/results"
log "start; interval=${INTERVAL}s"
while ! all_done; do
  log "tick"
  # Backbone artefacts
  pull "$REMOTE/runs/${NAME}_losses.csv"          "$LOCAL/runs/${NAME}_losses.csv"          100
  pull "$REMOTE/runs/${NAME}_attn_amplitude.csv"  "$LOCAL/runs/${NAME}_attn_amplitude.csv"  100
  pull "$REMOTE/runs/${NAME}_best_gap.pth"        "$LOCAL/runs/${NAME}_best_gap.pth"        80000000
  pull "$REMOTE/runs/${NAME}_best_loss.pth"       "$LOCAL/runs/${NAME}_best_loss.pth"       80000000
  pull "$REMOTE/runs/${NAME}_final.pth"           "$LOCAL/runs/${NAME}_final.pth"           80000000
  pull "$REMOTE/runs/${NAME}_FINAL.pth"           "$LOCAL/runs/${NAME}_FINAL.pth"           80000000
  # Per-cell eval results
  for c in "${CELLS[@]}"; do
    d="results/gift_eval_full_$c"
    mkdir -p "$LOCAL/$d"
    pull "$REMOTE/$d/all_results.csv" "$LOCAL/$d/all_results.csv" 100
    pull "$REMOTE/$d/summary.txt"     "$LOCAL/$d/summary.txt"     10
  done
  # Run log
  pull "$REMOTE/results/run_${NAME}.log" "$LOCAL/results/run_${NAME}.log" 100
  log "--- sleeping ${INTERVAL}s ---"
  sleep "$INTERVAL"
done
log "all 4 arm-3 summary.txt files present locally — sync exiting"
