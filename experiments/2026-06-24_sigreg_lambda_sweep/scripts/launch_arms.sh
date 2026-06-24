#!/bin/bash
# #363 — sweep driver: train each λ-sweep arm's backbone and then run its
# downstream cells. Arms run sequentially on the same GPU pair; per-arm
# downstream uses both GPUs in parallel (2L on GPU 0, 6L on GPU 1).
#
# Issue #363 §Objective order:
#   1) λ_e=10.0, λ_h=0.1   emb100_enc01
#   2) λ_e=10.0, λ_h=1.0   emb100_enc10
#   3) λ_e=10.0, λ_h=10.0  emb100_enc100
#
# The fourth arm (λ_e=1.0, λ_h=1.0, suffix emb10_enc10) is enabled only if
# RUN_OPTIONAL=1 — the issue ships it as "only if compute remains after the
# first three and the trajectory suggests an interior point is informative".
#
#   launch_arms.sh           run the three required arms
#   RUN_OPTIONAL=1 launch_arms.sh   also run emb10_enc10
#   ONLY="emb100_enc10 emb100_enc100" launch_arms.sh   pick a subset
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-363}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-24_sigreg_lambda_sweep}"
GPU="${GPU:-0}"
export WT OUT
BB_SCRIPT="$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/train_backbone_sigreg.sh"
DL_SCRIPT="$WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/launch_downstream.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sweep] $*"; }

# (lambda_e, lambda_h, suffix) in issue order — emb100_enc01 first, then
# emb100_enc10, then emb100_enc100. Optional fourth: emb10_enc10.
REQUIRED=(
  "10.0 0.1 emb100_enc01"
  "10.0 1.0 emb100_enc10"
  "10.0 10.0 emb100_enc100"
)
OPTIONAL=(
  "1.0 1.0 emb10_enc10"
)

ARMS=("${REQUIRED[@]}")
if [ "${RUN_OPTIONAL:-0}" = "1" ]; then
  ARMS=("${ARMS[@]}" "${OPTIONAL[@]}")
fi

run_arm(){ # lambda_e lambda_h suffix
  local le="$1" lh="$2" suffix="$3"
  log "ARM ${suffix} λ_e=${le} λ_h=${lh} — backbone start"
  bash "$BB_SCRIPT" "$GPU" "$le" "$lh" "$suffix" >>"$RES/sweep_bb_${suffix}.log" 2>&1
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
  read -r le lh suffix <<< "$entry"
  if [ -n "${ONLY:-}" ] && ! echo " $ONLY " | grep -q " $suffix "; then
    log "ARM ${suffix} skipped by ONLY filter"
    continue
  fi
  run_arm "$le" "$lh" "$suffix" || failed=$((failed+1))
done
log "sweep done; failed arms: $failed"
exit "$failed"
