#!/bin/bash
# #322 — single-GPU orchestrator (GPU 1). Trains the 5 batch-1024 backbones
# sequentially, each a single process @1024 with the checkpointed GRU. β arms first
# (fast, ~70 min — validate the full pipeline early), then the all-time arms
# (~12 h each, the (B·T)² Gram). Idempotent (skips FINAL-complete arms) + resumable.
# Gates on GPU 1 having room (~21 GB reserved peak measured) so a foreign tenant on
# GPU 1 can't OOM us. GPU 0 is left to its permanent foreign Jupyter kernels.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RES="$OUT/results"; mkdir -p "$RES"
OLOG="$RES/orchestrate.log"
STEPS="${STEPS:-12500}"; SAVE_EVERY="${SAVE_EVERY:-2500}"; ALLT_CHUNK="${ALLT_CHUNK:-2}"
GPU="${GPU:-1}"; GATE_MIB="${GATE_MIB:-22000}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($1+1))p"; }
wait_for_gpu(){
  local waited=0
  until [ "$(free_mib "$GPU")" -ge "$GATE_MIB" ]; do
    [ $((waited % 600)) -eq 0 ] && log "WAIT gpu$GPU free=$(free_mib "$GPU") MiB (need ${GATE_MIB})"
    sleep 120; waited=$((waited+120))
  done
}

#         arm      mix         tag           chunk
ARMS=(
  "beta     0.0078125   forked2       2"
  "beta     0.10        forked10pct   2"
  "alltime  0.5         forked        $ALLT_CHUNK"
  "alltime  0.10        forked10pct   $ALLT_CHUNK"
  "alltime  0.0078125   forked2       $ALLT_CHUNK"
)
log "ORCHESTRATOR(1gpu) start: GPU$GPU steps=$STEPS save_every=$SAVE_EVERY allt_chunk=$ALLT_CHUNK"
for spec in "${ARMS[@]}"; do
  set -- $spec; arm="$1"; mix="$2"; tag="$3"; chunk="$4"
  log ">>> ARM $arm $tag (mix=$mix chunk=$chunk steps=$STEPS)"
  wait_for_gpu
  bash "$HERE/train_backbone_b1024_1gpu.sh" "$arm" "$mix" "$tag" "$STEPS" "$SAVE_EVERY" "$chunk" "$GPU"
  log "<<< ARM $arm $tag rc=$?"
done
log "ORCHESTRATOR(1gpu) DONE (all 5 arms attempted)"
touch "$RES/.backbones_done"
