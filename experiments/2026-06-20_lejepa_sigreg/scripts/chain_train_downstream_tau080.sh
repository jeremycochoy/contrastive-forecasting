#!/bin/bash
# Wait for the τ=0.8 BB FINAL.pth then run downstream sequentially on GPU 1
# (GPU 0 is held by the parallel τ=0.9 arm).
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-357}"
OUT="${OUT:-/tmp/contrastive-forecasting-357/reports/2026-06-21_lejepa_sigreg_tau098}"
export WT OUT
NAME=bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau080
BB="$OUT/runs/${NAME}_FINAL.pth"
SCRIPT="$WT/experiments/2026-06-20_lejepa_sigreg/scripts/downstream_sigreg_tau080.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain-tau080] $*"; }

log "waiting for BB at $BB"
while [ ! -f "$BB" ]; do sleep 60; done
log "BB present ($(du -h "$BB" | cut -f1)); running downstream sequentially on GPU 1"

bash "$SCRIPT" 2 1 >>"$RES/dl_2L_tau080.log" 2>&1; rc2=$?
log "2L finished rc=$rc2; starting 6L on GPU 1"
bash "$SCRIPT" 6 1 >>"$RES/dl_6L_tau080.log" 2>&1; rc6=$?
log "6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
