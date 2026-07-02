#!/bin/bash
# #366 — orchestrator for two NEW arms (C, D) at τ=0.90 to extend the
# (λ_e, λ_h) matrix at the cross's τ. Identical recipe to A/B; only λ
# differs.
#   Arm C — λ_e=1,   λ_h=1,   τ=0.90  (suffix lC_emb10_enc10_tau090)
#   Arm D — λ_e=10,  λ_h=10,  τ=0.90  (suffix lD_emb100_enc100_tau090)
# Expects WT, OUT exported. Runs both BB in parallel on GPUs 0/1, then
# downstream sequentially (per-arm 2L on g0 + 6L on g1 in parallel).
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
EXP="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts"
BB="$EXP/train_backbone_sigreg.sh"
DL="$EXP/launch_downstream.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [arms-cd] $*"; }

SC=lC_emb10_enc10_tau090
SD=lD_emb100_enc100_tau090

if [ "${SKIP_BB:-0}" != 1 ]; then
  log "BB phase: C (λ_e=1, λ_h=1) on g0; D (λ_e=10, λ_h=10) on g1 — parallel"
  bash "$BB" 0 1   1   0.90 "$SC" >>"$RES/sweep_bb_${SC}.log" 2>&1 & pidC=$!
  bash "$BB" 1 10  10  0.90 "$SD" >>"$RES/sweep_bb_${SD}.log" 2>&1 & pidD=$!
  log "BB pids: C=$pidC  D=$pidD"
  wait $pidC; rcC=$?
  wait $pidD; rcD=$?
  log "BB done: C rc=$rcC  D rc=$rcD"
  [ $rcC -eq 0 ] && [ $rcD -eq 0 ] || { log "ABORT: backbone failure"; exit 1; }
fi

log "DL phase: C (2L on g0, 6L on g1 — parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SC" >>"$RES/sweep_dl_${SC}.log" 2>&1
rcDC=$?
log "DL C done rc=$rcDC"
log "DL phase: D (2L on g0, 6L on g1 — parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SD" >>"$RES/sweep_dl_${SD}.log" 2>&1
rcDD=$?
log "DL D done rc=$rcDD"
log "all phases complete; C rc=$rcDC D rc=$rcDD"
exit $(( rcDC + rcDD ))
