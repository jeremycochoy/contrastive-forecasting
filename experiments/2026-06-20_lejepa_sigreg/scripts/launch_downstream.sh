#!/bin/bash
# Run downstream q-head + GIFT-Eval for the SIGReg backbone on both GPUs concurrently.
# 2L on GPU 0, 6L on GPU 1. Same protocol as #353 downstream_ema launcher.
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-355}"
OUT="${OUT:-/tmp/contrastive-forecasting-355/reports/2026-06-20_lejepa_sigreg}"
export WT OUT
SCRIPT="$WT/experiments/2026-06-20_lejepa_sigreg/scripts/downstream_sigreg.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-launch] $*"; }

log "starting 2L on GPU 0 + 6L on GPU 1 (parallel)"
bash "$SCRIPT" 2 0 >>"$RES/dl_2L.log" 2>&1 &
pid2=$!
bash "$SCRIPT" 6 1 >>"$RES/dl_6L.log" 2>&1 &
pid6=$!
log "PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "2L finished rc=$rc2  6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
