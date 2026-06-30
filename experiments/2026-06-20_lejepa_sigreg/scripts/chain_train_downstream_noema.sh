#!/bin/bash
# Wait for the no-EMA BB FINAL.pth then launch downstream on both GPUs in parallel.
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-357}"
OUT="${OUT:-/tmp/contrastive-forecasting-357/reports/2026-06-21_lejepa_sigreg_tau098}"
export WT OUT
NAME=bb_allt08_xftrip_nobn_enc3_sigreg_qk_aon_b512_cpc_noema
BB="$OUT/runs/${NAME}_FINAL.pth"
SCRIPT="$WT/experiments/2026-06-20_lejepa_sigreg/scripts/downstream_sigreg_noema.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain-noema] $*"; }

log "waiting for BB at $BB"
while [ ! -f "$BB" ]; do sleep 60; done
log "BB present ($(du -h "$BB" | cut -f1)); launching downstream on both GPUs"

bash "$SCRIPT" 2 0 >>"$RES/dl_2L_noema.log" 2>&1 &
pid2=$!
bash "$SCRIPT" 6 1 >>"$RES/dl_6L_noema.log" 2>&1 &
pid6=$!
log "PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "2L finished rc=$rc2  6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
