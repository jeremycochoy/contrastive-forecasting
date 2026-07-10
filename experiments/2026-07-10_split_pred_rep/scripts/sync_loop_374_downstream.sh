#!/bin/bash
# #374 — 15-min sync loop for the vast.ai downstream (2L + 6L cells).
# Pulls all_results.csv + summary.txt for each of the 4 (arm, ckpt) cells,
# plus the two qhead FINAL checkpoints. Exits when all four summary.txt
# files are present locally.
set -uo pipefail
HOST=ssh3.vast.ai
PORT=38448
REMOTE=/workspace/cf-374/experiments/2026-07-10_split_pred_rep
LOCAL=/tmp/contrastive-forecasting-374/experiments/2026-07-10_split_pred_rep
SAFE_PULL=/tmp/contrastive-forecasting-374/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh
INTERVAL=900
TAG=split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090
CELLS=("${TAG}_2L" "${TAG}_last_2L" "${TAG}_6L" "${TAG}_last_6L")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sync-dl] $*"; }

all_done(){
  for c in "${CELLS[@]}"; do
    [ -f "$LOCAL/results/gift_eval_full_$c/summary.txt" ] || return 1
  done
  return 0
}

pull(){ bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$2" "$3" || echo "  (skip: $1)"; }

log "start; interval=${INTERVAL}s; watching ${#CELLS[@]} cells"

while ! all_done; do
  log "tick"
  for c in "${CELLS[@]}"; do
    d="results/gift_eval_full_$c"
    mkdir -p "$LOCAL/$d"
    pull "$REMOTE/$d/all_results.csv" "$LOCAL/$d/all_results.csv" 100
    pull "$REMOTE/$d/summary.txt"     "$LOCAL/$d/summary.txt"     10
  done
  # Also grab the qhead last FINAL when it lands, so downstream reruns
  # aren't needed on elisa.
  for hl in 2 6; do
    for name in "qhead_${hl}L_${TAG}_last_FINAL" "qhead_${hl}L_${TAG}_last_best"; do
      pull "$REMOTE/runs/${name}.pth" "$LOCAL/runs/${name}.pth" 8000000
    done
  done
  pull "$REMOTE/../../../tmp/vast_dl_launch.log" "$LOCAL/results/vast_dl_launch.log" 10
  log "--- sleeping ${INTERVAL}s ---"
  sleep "$INTERVAL"
done
log "all ${#CELLS[@]} summary.txt files present locally — sync exiting"
