#!/bin/bash
# #371 — sequentially run every remaining head/eval cell on ONE GPU.
# Order: retry the two OOM'd cells (2L cells all completed; 6L step20000
# then step25000) then the six extension cells (30000/35000/37500 × 2L/6L).
# Skips cells whose FINAL.pth + summary.txt already exist. Waits until the
# backbone step<N> checkpoint is on disk before that cell's cycle starts.
#
#   queue_remaining.sh <gpu>
set -uo pipefail
GPU="${1:?gpu}"
: "${WT:?}"; : "${EXP:?}"; : "${SYNC:?}"
LOG="$EXP/results/queue_remaining_gpu${GPU}.log"
NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [queue g$GPU] $*" | tee -a "$LOG"; }

wait_bb(){ # STEP
  local step="$1"
  for _ in $(seq 1 720); do  # 720 * 60 s = 12 h ceiling
    for cand in "$SYNC/${NAME}_step${step}.pth" "$SYNC/${NAME}_r"*"_step${step}.pth"; do
      [ -f "$cand" ] && return 0
    done
    sleep 60
  done
  log "TIMEOUT waiting for step${step} backbone"
  return 1
}

# (HL, STEP) pairs. 6L retry first (was left broken), then paired extension cells.
# HL is the layer-count int (2 or 6); dl_one_cell.sh appends the trailing 'L'.
CELLS=( "6 20000" "6 25000" "2 30000" "6 30000" "2 35000" "6 35000" "2 37500" "6 37500" )
for cell in "${CELLS[@]}"; do
  hl=${cell% *}; step=${cell#* }
  wait_bb "$step" || continue
  log "RUN cell HL=$hl step=$step"
  WT="$WT" EXP="$EXP" SYNC="$SYNC" bash "$EXP/scripts/dl_one_cell.sh" "$hl" "$GPU" "$step" \
    >>"$LOG" 2>&1 && log "OK  cell HL=$hl step=$step" || log "FAIL cell HL=$hl step=$step"
done
log "queue done"
