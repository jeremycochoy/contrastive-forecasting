#!/bin/bash
# Wait for backbone FINAL.pth then launch downstream (both GPUs).
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-357}"
OUT="${OUT:-/tmp/contrastive-forecasting-357/reports/2026-06-21_lejepa_sigreg_tau098}"
export WT OUT
NAME=bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau098
BB="$OUT/runs/${NAME}_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain-tau098] $*"; }

log "waiting for BB at $BB"
while [ ! -f "$BB" ]; do sleep 60; done
log "BB present ($(du -h "$BB" | cut -f1)); launching downstream"

bash "$WT/experiments/2026-06-20_lejepa_sigreg/scripts/launch_downstream_tau098.sh"
rc=$?
log "downstream finished rc=$rc"
exit $rc
