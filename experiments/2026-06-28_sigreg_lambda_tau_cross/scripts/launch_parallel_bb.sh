#!/bin/bash
# #366 — two-GPU parallel orchestrator: both backbones in parallel (one per
# GPU), then downstream per arm sequentially using both GPUs internally.
# Halves wall-clock vs launch_arms.sh's sequential arm loop.
#
# Expects WT, OUT exported. Reads winners.sh from $OUT.
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
EXP="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts"
BB="$EXP/train_backbone_sigreg.sh"
DL="$EXP/launch_downstream.sh"
WIN="$OUT/winners.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [par-orch] $*"; }

[ -f "$WIN" ] || { log "ABORT: $WIN missing"; exit 2; }
# shellcheck disable=SC1090
. "$WIN"
for v in ARM_A_LAMBDA_E ARM_A_LAMBDA_H ARM_B_LAMBDA_E ARM_B_LAMBDA_H BEST_TAU; do
  [ -n "${!v:-}" ] || { log "ABORT: $v unset"; exit 2; }
done
suffix_for(){ awk -v p="$1" -v le="$2" -v lh="$3" -v t="$4" \
  'BEGIN { printf "%s_emb%.0f_enc%.0f_tau%03.0f\n", p, le*10, lh*10, t*100 }'; }
SA=$(suffix_for lA "$ARM_A_LAMBDA_E" "$ARM_A_LAMBDA_H" "$BEST_TAU")
SB=$(suffix_for lB "$ARM_B_LAMBDA_E" "$ARM_B_LAMBDA_H" "$BEST_TAU")
log "Arm A: λ_e=${ARM_A_LAMBDA_E} λ_h=${ARM_A_LAMBDA_H} τ=${BEST_TAU} suffix=${SA}"
log "Arm B: λ_e=${ARM_B_LAMBDA_E} λ_h=${ARM_B_LAMBDA_H} τ=${BEST_TAU} suffix=${SB}"

if [ "${SKIP_BB:-0}" != 1 ]; then
  log "BB phase: A on GPU 0, B on GPU 1 in parallel"
  bash "$BB" 0 "$ARM_A_LAMBDA_E" "$ARM_A_LAMBDA_H" "$BEST_TAU" "$SA" \
    >>"$RES/sweep_bb_${SA}.log" 2>&1 & pidA=$!
  bash "$BB" 1 "$ARM_B_LAMBDA_E" "$ARM_B_LAMBDA_H" "$BEST_TAU" "$SB" \
    >>"$RES/sweep_bb_${SB}.log" 2>&1 & pidB=$!
  log "BB pids: A=$pidA  B=$pidB"
  wait $pidA; rcA=$?
  wait $pidB; rcB=$?
  log "BB done: A rc=$rcA  B rc=$rcB"
  if [ $rcA -ne 0 ] || [ $rcB -ne 0 ]; then
    log "ABORT: a backbone failed; not running downstream"
    exit 1
  fi
fi

log "DL phase: A first (2L on GPU 0, 6L on GPU 1, parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SA" >>"$RES/sweep_dl_${SA}.log" 2>&1
rcDA=$?
log "DL A done rc=$rcDA"
log "DL phase: B (2L on GPU 0, 6L on GPU 1, parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SB" >>"$RES/sweep_dl_${SB}.log" 2>&1
rcDB=$?
log "DL B done rc=$rcDB"

log "all phases complete; rc A=${rcA:-skip} B=${rcB:-skip} DL_A=$rcDA DL_B=$rcDB"
exit $(( (rcA != 0) + (rcB != 0) + (rcDA != 0) + (rcDB != 0) ))
