#!/bin/bash
# #325 — elisa orchestrator (GPU 1). Trains the crossfade backbone, then the 2L and 6L
# downstream q-heads + GIFT-Eval (triage + full). Serial on GPU 1 (GPU 0 is held by
# foreign Jupyter kernels). Idempotent + resumable (per-step FINAL-exists skips).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10}"
RES="$OUT/results"; mkdir -p "$RES"
OLOG="$RES/orchestrate.log"
STEPS="${STEPS:-12500}"; SAVE_EVERY="${SAVE_EVERY:-2500}"; ALLT_CHUNK="${ALLT_CHUNK:-2}"
GPU="${GPU:-1}"; GATE_MIB="${GATE_MIB:-22000}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($1+1))p"; }
wait_for_gpu(){ local need="$1" waited=0
  until [ "$(free_mib "$GPU")" -ge "$need" ]; do
    [ $((waited % 600)) -eq 0 ] && log "WAIT gpu$GPU free=$(free_mib "$GPU") MiB (need ${need})"
    sleep 120; waited=$((waited+120)); done; }

log "ORCHESTRATOR start: GPU$GPU steps=$STEPS save_every=$SAVE_EVERY allt_chunk=$ALLT_CHUNK"
log ">>> BACKBONE allt·10% + crossfade·10%"
wait_for_gpu "$GATE_MIB"
bash "$HERE/train_backbone_crossfade.sh" "$STEPS" "$SAVE_EVERY" "$ALLT_CHUNK" "$GPU"
log "<<< BACKBONE rc=$?"

for HL in 2 6; do
  log ">>> DOWNSTREAM ${HL}L"
  wait_for_gpu 12000
  bash "$HERE/downstream_crossfade.sh" "$HL" "$GPU" "triage full"
  log "<<< DOWNSTREAM ${HL}L rc=$?"
done
log "ORCHESTRATOR DONE"
touch "$RES/.all_done"
