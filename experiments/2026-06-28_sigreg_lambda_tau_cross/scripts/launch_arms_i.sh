#!/bin/bash
# #366 — Arm I: λ_e=100, λ_h=1, τ=0.90 (fills the emb1000 column on the
# λ_h=1 row).
#   suffix lI_emb1000_enc10_tau090
# Expects WT, OUT exported. Uses only GPU 0 for everything (BB, then 2L
# head, then 6L head — sequential) so GPU 1 stays free for a parallel
# follow-up run.
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
EXP="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts"
BB="$EXP/train_backbone_sigreg.sh"
DL="$EXP/launch_downstream.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [arm-i] $*"; }

SI=lI_emb1000_enc10_tau090

if [ "${SKIP_BB:-0}" != 1 ]; then
  log "BB phase: I (λ_e=100, λ_h=1) on g0"
  bash "$BB" 0 100 1 0.90 "$SI" >>"$RES/sweep_bb_${SI}.log" 2>&1
  rcI=$?
  log "BB done: I rc=$rcI"
  [ $rcI -eq 0 ] || { log "ABORT: backbone failure"; exit 1; }
fi

DL_SCRIPT="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts/downstream_sigreg.sh"

log "DL phase: I 2L on g0 (sequential; g1 stays free)"
bash "$DL_SCRIPT" 2 0 "$SI" >>"$RES/dl_2L_${SI}.log" 2>&1
rc2=$?
log "DL I 2L done rc=$rc2"

log "DL phase: I 6L on g0 (sequential)"
bash "$DL_SCRIPT" 6 0 "$SI" >>"$RES/dl_6L_${SI}.log" 2>&1
rc6=$?
log "DL I 6L done rc=$rc6"

log "all phases complete; 2L=$rc2 6L=$rc6"
exit $(( rc2 + rc6 ))
