#!/bin/bash
# Run downstream q-head + GIFT-Eval for the SIGReg-tau098 backbone.
# GPU 1 is held by another experiment, so 2L and 6L are run sequentially on GPU 0.
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-357}"
OUT="${OUT:-/tmp/contrastive-forecasting-357/reports/2026-06-21_lejepa_sigreg_tau098}"
export WT OUT
SCRIPT="$WT/experiments/2026-06-20_lejepa_sigreg/scripts/downstream_sigreg_tau098.sh"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-launch-tau098] $*"; }

log "starting 2L on GPU 0 (sequential)"
bash "$SCRIPT" 2 0 >>"$RES/dl_2L.log" 2>&1; rc2=$?
log "2L finished rc=$rc2; starting 6L on GPU 0"
bash "$SCRIPT" 6 0 >>"$RES/dl_6L.log" 2>&1; rc6=$?
log "6L finished rc=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
