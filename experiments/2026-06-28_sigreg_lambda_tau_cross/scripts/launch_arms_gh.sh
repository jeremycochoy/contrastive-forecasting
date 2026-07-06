#!/bin/bash
# #366 — Arms G, H at τ=0.90 (λ_h=10 row, off-diagonal cells).
#   Arm G — λ_e=100, λ_h=10, τ=0.90  (suffix lG_emb1000_enc100_tau090)
#   Arm H — λ_e=1,   λ_h=10, τ=0.90  (suffix lH_emb10_enc100_tau090)
# Expects WT, OUT exported.
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
EXP="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts"
BB="$EXP/train_backbone_sigreg.sh"
DL="$EXP/launch_downstream.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [arms-gh] $*"; }

SG=lG_emb1000_enc100_tau090
SH=lH_emb10_enc100_tau090

if [ "${SKIP_BB:-0}" != 1 ]; then
  log "BB phase: G (λ_e=100, λ_h=10) on g0; H (λ_e=1, λ_h=10) on g1 — parallel"
  bash "$BB" 0 100 10 0.90 "$SG" >>"$RES/sweep_bb_${SG}.log" 2>&1 & pidG=$!
  bash "$BB" 1 1   10 0.90 "$SH" >>"$RES/sweep_bb_${SH}.log" 2>&1 & pidH=$!
  log "BB pids: G=$pidG  H=$pidH"
  wait $pidG; rcG=$?
  wait $pidH; rcH=$?
  log "BB done: G rc=$rcG  H rc=$rcH"
  [ $rcG -eq 0 ] && [ $rcH -eq 0 ] || { log "ABORT: backbone failure"; exit 1; }
fi

log "DL phase: G (2L on g0, 6L on g1 — parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SG" >>"$RES/sweep_dl_${SG}.log" 2>&1
rcDG=$?
log "DL G done rc=$rcDG"
log "DL phase: H (2L on g0, 6L on g1 — parallel)"
GPU_2L=0 GPU_6L=1 bash "$DL" "$SH" >>"$RES/sweep_dl_${SH}.log" 2>&1
rcDH=$?
log "DL H done rc=$rcDH"
log "all phases complete; G rc=$rcDG H rc=$rcDH"
exit $(( rcDG + rcDH ))
