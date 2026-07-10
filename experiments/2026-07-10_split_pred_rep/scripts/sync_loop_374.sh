#!/bin/bash
# Persistent 15-min sync loop between the vast.ai instance and elisa.
# Pulls checkpoints, losses.csv, attn_amplitude.csv, and run.log using
# safe_pull.sh (atomic .tmp → mv, .prev backup on rotate, size floor).
# Stops when the FINAL.pth appears locally.
#
#   nohup setsid bash sync_loop_374.sh > sync_loop_374.log 2>&1 &
set -uo pipefail
HOST=ssh5.vast.ai
PORT=11008
REMOTE=/workspace/cf-374/experiments/2026-07-10_split_pred_rep
LOCAL=/tmp/contrastive-forecasting-374/experiments/2026-07-10_split_pred_rep
SAFE_PULL=/tmp/contrastive-forecasting-374/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh
INTERVAL=900
NAME=bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090
FINAL="$LOCAL/runs/${NAME}_FINAL.pth"

# Per-class minimum-byte floors (blanket floor eats small files — CLAUDE.md).
declare -A MIN=(
  [best_gap.pth]=80000000
  [best_gap_optimizer.pth]=120000000
  [best_loss.pth]=80000000
  [best_loss_optimizer.pth]=120000000
  [losses.csv]=100
  [attn_amplitude.csv]=100
)

pull(){ # remote_path local_path min_bytes
  bash "$SAFE_PULL" "$HOST" "$PORT" "$1" "$2" "$3" || echo "  (skip: $1 not present or too small)"
}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sync-374] $*"; }

log "start; interval=${INTERVAL}s; final sentinel=$FINAL"
mkdir -p "$LOCAL/runs" "$LOCAL/results"

# Loop until FINAL.pth arrives locally (which the training emits at 12,500).
while [ ! -f "$FINAL" ]; do
  log "tick"
  for f in best_gap.pth best_gap_optimizer.pth best_loss.pth best_loss_optimizer.pth; do
    pull "$REMOTE/runs/${NAME}_$f" "$LOCAL/runs/${NAME}_$f" "${MIN[$f]}"
  done
  for f in losses.csv attn_amplitude.csv; do
    pull "$REMOTE/runs/${NAME}_$f" "$LOCAL/runs/${NAME}_$f" "${MIN[$f]}"
  done
  # Pull periodic step checkpoints if they appear (2500k, 5000k, 7500k, 10000k, 12500k).
  for step in 2500 5000 7500 10000 12500; do
    pull "$REMOTE/runs/${NAME}_${step}k.pth" "$LOCAL/runs/${NAME}_${step}k.pth" 80000000
    pull "$REMOTE/runs/${NAME}_${step}k_optimizer.pth" "$LOCAL/runs/${NAME}_${step}k_optimizer.pth" 120000000
  done
  # Try to grab _final and _FINAL (end-of-train products).
  pull "$REMOTE/runs/${NAME}_final.pth" "$LOCAL/runs/${NAME}_final.pth" 80000000
  pull "$REMOTE/runs/${NAME}_final_optimizer.pth" "$LOCAL/runs/${NAME}_final_optimizer.pth" 120000000
  pull "$REMOTE/runs/${NAME}_FINAL.pth" "$FINAL" 80000000
  pull "$REMOTE/results/run_${NAME}.log" "$LOCAL/results/run_${NAME}.log" 100
  log "--- sleeping ${INTERVAL}s ---"
  sleep "$INTERVAL"
done
log "FINAL sentinel present at $FINAL — sync loop exiting"
