#!/bin/bash
# #318 — unattended eval pipeline. Waits for the xshh backbone to finish AND
# release the GPU, then runs the full eval matrix serially on one GPU (GPU 1 is
# used by a concurrent experiment, so everything runs on GPU 0). Priority
# order: xshh headline first, then β 6L, then GM-vs-step. Idempotent —
# downstream.sh skips any q-head/eval whose output already exists, so this is
# safe to re-launch.
#
#   gpu0_pipeline.sh [gpu]
set -uo pipefail
GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="$HERE/downstream.sh"
MINE=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh/runs
BETA=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound/runs
FINAL="$MINE/bb_xshh_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [pipeline g$GPU] $*"; }

log "waiting for backbone FINAL + GPU release ..."
until [ -f "$FINAL" ] && ! pgrep -fa "train.py" | grep -q "bb_xshh_50k"; do sleep 60; done
log "backbone ready ($FINAL); starting eval matrix"

# --- P1 headline (final backbone, both heads, triage+full) ---
bash "$DS" "$FINAL"                  2 xshh_50k "$GPU" "triage full"
bash "$DS" "$FINAL"                  6 xshh_50k "$GPU" "triage full"
bash "$DS" "$BETA/bb_beta_50k_FINAL.pth" 6 beta_50k "$GPU" "triage full"   # β 2L cited from #309

# --- P2 GM-MASE vs training step (full-97, 2L) ---
for k in 20k 35k; do
  [ -f "$MINE/bb_xshh_50k_${k}.pth" ] && bash "$DS" "$MINE/bb_xshh_50k_${k}.pth" 2 "xshh_${k}" "$GPU" "full" || log "missing mine $k"
  [ -f "$BETA/bb_beta_50k_${k}.pth" ] && bash "$DS" "$BETA/bb_beta_50k_${k}.pth" 2 "beta_${k}" "$GPU" "full" || log "missing beta $k"
done

log "=== EVAL_MATRIX_DONE ==="
