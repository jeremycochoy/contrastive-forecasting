#!/bin/bash
# #350 — run the full bilinear-arm downstream once the backbone FINAL exists:
# 2L head pipeline on GPU_A and 6L on GPU_B in parallel (each does best-loss +
# last backbone, then GIFT-Eval full-97), then analysis + plots. Idempotent:
# every stage skips when its FINAL / summary already exists, so re-running after
# a crash resumes.
#   run_downstream.sh [gpu_2L] [gpu_6L]
set -uo pipefail
GPU2="${1:-1}"; GPU6="${2:-0}"
WT="/home/jupyter/cf-wt-350-bilinear"
OUT="/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss"
DIR="$WT/experiments/2026-06-16_bilinear_main_loss/scripts"
BB="$OUT/runs/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [downstream] $*"; }
[ -f "$BB" ] || { log "ABORT: backbone FINAL missing ($BB)"; exit 1; }
log "2L on GPU$GPU2, 6L on GPU$GPU6 (parallel)"
WT="$WT" OUT="$OUT" bash "$DIR/downstream.sh" bilinear 2 "$GPU2" >"$OUT/results/dl_2L.log" 2>&1 &
P2=$!
WT="$WT" OUT="$OUT" bash "$DIR/downstream.sh" bilinear 6 "$GPU6" >"$OUT/results/dl_6L.log" 2>&1 &
P6=$!
wait $P2; r2=$?; wait $P6; r6=$?
log "2L rc=$r2  6L rc=$r6"
log "analysis"
python3 "$DIR/analyze_bilinear.py" >"$OUT/results/analysis.out" 2>&1 || log "analyze failed"
PYTHONPATH="$WT" python3 "$DIR/analyze_W.py" >"$OUT/results/analyze_W.out" 2>&1 || log "analyze_W failed"
python3 "$DIR/plot_results.py" >>"$OUT/results/analysis.out" 2>&1 || log "plot_results failed"
python3 "$DIR/plot_training_dynamics.py" >>"$OUT/results/analysis.out" 2>&1 || log "plot_dyn failed"
log "downstream pipeline complete"
