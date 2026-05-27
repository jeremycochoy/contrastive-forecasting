#!/bin/bash
# #318 eval matrix — drives downstream.sh over all cells, in priority order.
# IDEMPOTENT and resumable: re-running skips completed q-heads/evals. Safe to
# run two instances on two GPUs (different cells complete independently).
#
#   run_eval_matrix.sh <gpu> [phase]
#     gpu     CUDA_VISIBLE_DEVICES (0|1)
#     phase   p1 (headline only) | p2 (vs-step only) | all (default)
#
# Backbones:
#   mine  = OUT/runs/bb_xshh_50k_{FINAL,20k,35k}.pth
#   beta  = #309 bb_beta_50k_{FINAL,20k,35k}.pth (read-only; β=1.3272 ref)
set -uo pipefail
GPU="${1:?gpu}"; PHASE="${2:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="$HERE/downstream.sh"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh
MINE="$OUT/runs"
BETA=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound/runs
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [matrix g$GPU $PHASE] $*"; }

cell(){ bash "$DS" "$@" "$GPU" "${5:-triage full}"; }   # backbone head_layers tag <sets>

# --- Phase 1: headline (final backbones, both heads, triage+full) ---
p1(){
  log "=== P1 headline ==="
  cell "$MINE/bb_xshh_50k_FINAL.pth" 2 xshh_50k "" "triage full"
  cell "$MINE/bb_xshh_50k_FINAL.pth" 6 xshh_50k "" "triage full"
  cell "$BETA/bb_beta_50k_FINAL.pth" 6 beta_50k "" "triage full"   # β 6L (new); β 2L cited from #309=1.3272
}

# --- Phase 2: GM-MASE vs training step (full-97, 2L head) ---
p2(){
  log "=== P2 vs-step (full-97, 2L) ==="
  for k in 20k 35k; do
    [ -f "$MINE/bb_xshh_50k_${k}.pth" ] && cell "$MINE/bb_xshh_50k_${k}.pth" 2 "xshh_${k}" "" "full" || log "missing mine $k"
    [ -f "$BETA/bb_beta_50k_${k}.pth" ] && cell "$BETA/bb_beta_50k_${k}.pth" 2 "beta_${k}" "" "full" || log "missing beta $k"
  done
  # 50k full @ 2L: mine from P1 cell; beta from #309 (gift_eval_full_bb_beta_50k=1.3272).
}

case "$PHASE" in
  p1) p1 ;;
  p2) p2 ;;
  all) p1; p2 ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac
log "=== matrix phase '$PHASE' complete ==="
