#!/bin/bash
# #366 — orchestrator for Arms E, F at τ=0.90; extends the (λ_e, λ_h)
# matrix on the diagonal at λ_e=λ_h.
#   Arm E — λ_e=100,  λ_h=100,  τ=0.90  (suffix lE_emb1000_enc1000_tau090)
#   Arm F — λ_e=1000, λ_h=1000, τ=0.90  (suffix lF_emb10000_enc10000_tau090)
# Expects WT, OUT exported. Identical layout to launch_arms_cd.sh.
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
EXP="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts"
BB="$EXP/train_backbone_sigreg.sh"
DL="$EXP/launch_downstream.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [arms-ef] $*"; }

SE=lE_emb1000_enc1000_tau090
SF=lF_emb10000_enc10000_tau090

if [ "${SKIP_BB:-0}" != 1 ]; then
  log "BB phase: E (λ_e=100, λ_h=100) on g0; F (λ_e=1000, λ_h=1000) on g1 — parallel"
  bash "$BB" 0 100  100  0.90 "$SE" >>"$RES/sweep_bb_${SE}.log" 2>&1 & pidE=$!
  bash "$BB" 1 1000 1000 0.90 "$SF" >>"$RES/sweep_bb_${SF}.log" 2>&1 & pidF=$!
  log "BB pids: E=$pidE  F=$pidF"
  wait $pidE; rcE=$?
  wait $pidF; rcF=$?
  log "BB done: E rc=$rcE  F rc=$rcF"
  [ $rcE -eq 0 ] && [ $rcF -eq 0 ] || { log "ABORT: backbone failure"; exit 1; }
fi

log "DL phase: E (2L on g0, 6L on g1 — parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SE" >>"$RES/sweep_dl_${SE}.log" 2>&1
rcDE=$?
log "DL E done rc=$rcDE"
log "DL phase: F (2L on g0, 6L on g1 — parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SF" >>"$RES/sweep_dl_${SF}.log" 2>&1
rcDF=$?
log "DL F done rc=$rcDF"
log "all phases complete; E rc=$rcDE F rc=$rcDF"
exit $(( rcDE + rcDF ))
