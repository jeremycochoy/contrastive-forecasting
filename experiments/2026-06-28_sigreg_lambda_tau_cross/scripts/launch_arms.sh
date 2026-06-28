#!/bin/bash
# #366 — sweep driver: train each cross arm's backbone and then run its
# downstream cells. Arms run sequentially on the same GPU pair; per-arm
# downstream uses both GPUs in parallel (2L on GPU 0, 6L on GPU 1).
#
# Two arms (issue #366 §Objective):
#   Arm A — λ_e=10.0,   λ_h=1.0, τ=0.90   (lA_emb100_enc10_tau090)
#     #363 best-at-best λ pair × #357 best τ.
#   Arm B — λ_e=1000.0, λ_h=1.0, τ=0.90   (lB_emb10000_enc10_tau090)
#     #363 best-at-last λ pair × #357 best τ.
#
#   launch_arms.sh                     run both arms
#   ONLY="lA_emb100_enc10_tau090" launch_arms.sh   pick a subset
set -uo pipefail
: "${WT:?WT (worktree root containing experiments/<this-dir>/scripts/...) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set; e.g. OUT=\$WT/experiments/2026-06-28_sigreg_lambda_tau_cross}"
GPU="${GPU:-0}"
export WT OUT
[ -d "$WT" ] || { echo "[cross] ABORT: WT does not exist: $WT" >&2; exit 2; }
BB_SCRIPT="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts/train_backbone_sigreg.sh"
DL_SCRIPT="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts/launch_downstream.sh"
[ -f "$BB_SCRIPT" ] || { echo "[cross] ABORT: BB_SCRIPT not found: $BB_SCRIPT" >&2; exit 2; }
[ -f "$DL_SCRIPT" ] || { echo "[cross] ABORT: DL_SCRIPT not found: $DL_SCRIPT" >&2; exit 2; }
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [cross] $*"; }

# (lambda_e, lambda_h, tau, suffix) — Arm A then Arm B per the issue order.
ARMS=(
  "10.0   1.0 0.90 lA_emb100_enc10_tau090"
  "1000.0 1.0 0.90 lB_emb10000_enc10_tau090"
)

run_arm(){ # lambda_e lambda_h tau suffix
  local le="$1" lh="$2" tau="$3" suffix="$4"
  log "ARM ${suffix} λ_e=${le} λ_h=${lh} τ=${tau} — backbone start"
  bash "$BB_SCRIPT" "$GPU" "$le" "$lh" "$tau" "$suffix" >>"$RES/sweep_bb_${suffix}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "ARM ${suffix} backbone FAILED rc=$rc — skipping downstream"
    return $rc
  fi
  log "ARM ${suffix} — downstream start"
  bash "$DL_SCRIPT" "$suffix" >>"$RES/sweep_dl_${suffix}.log" 2>&1
  rc=$?
  log "ARM ${suffix} downstream rc=$rc"
  return $rc
}

failed=0
for entry in "${ARMS[@]}"; do
  read -r le lh tau suffix <<< "$entry"
  if [ -n "${ONLY:-}" ] && ! echo " $ONLY " | grep -q " $suffix "; then
    log "ARM ${suffix} skipped by ONLY filter"
    continue
  fi
  run_arm "$le" "$lh" "$tau" "$suffix" || failed=$((failed+1))
done
log "cross done; failed arms: $failed"
exit "$failed"
