#!/bin/bash
# #369 follow-up — sequential head+eval queue for the extended
# checkpoints. Runs a fixed list of `<HL>@<step>` cells on ONE GPU,
# waiting for each BB trajectory file to exist before dispatching.
# Skips cells whose eval summary.txt already exists (idempotent restart).
#
#   dl_chain.sh <gpu> <cell> [<cell> ...]
# e.g.
#   dl_chain.sh 0 6L@15000 2L@20000 6L@20000 2L@25000 6L@25000
set -uo pipefail
: "${WT:?}"; : "${OUT:?}"
GPU="${1:?gpu}"; shift
SUFFIX="${SUFFIX:-l_emb10_enc10_tau090_b1024}"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_${SUFFIX}"
RUNS="$OUT/runs"; RES="$OUT/results"
DL="$OUT/scripts/dl_at_step.sh"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-chain g$GPU] $*"; }
log "queue: $*"

for cell in "$@"; do
  HL="${cell%%@*}"; HL="${HL%L}"
  STEP="${cell##*@}"
  # Skip if already scored
  done_mark="$RES/gift_eval_full_${TAG}_step${STEP}_${HL}L/summary.txt"
  if [ -f "$done_mark" ]; then
    log "SKIP ${cell} (already scored — $done_mark)"
    continue
  fi
  # Block until the BB checkpoint at STEP lands
  while :; do
    if [ -f "$RUNS/bb_${TAG}_step${STEP}.pth" ]; then break; fi
    if ls "$RUNS/bb_${TAG}"_r*_step"${STEP}".pth 2>/dev/null >/dev/null; then break; fi
    log "WAIT for bb_..._step${STEP}.pth"
    sleep 300
  done
  log "RUN ${cell} on GPU ${GPU}"
  bash "$DL" "$HL" "$GPU" "$SUFFIX" "$STEP"
  rc=$?
  log "DONE ${cell} rc=$rc"
  if [ "$rc" -ne 0 ]; then
    log "ABORT chain due to failing cell ${cell}"; exit "$rc"
  fi
done
log "chain complete"
