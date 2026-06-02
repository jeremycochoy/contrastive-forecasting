#!/bin/bash
# #322 — STABILISED 5-arm sweep at the fix: QK-norm + attention-output RMSNorm, LR 1e-3.
# (b1024 collapses without it; this config clears the collapse zone — converges like b256.)
# Runs the 5 arms single-GPU @1024 on GPU1, sequential, gated on GPU1 free, idempotent
# (skips FINAL-complete arms — so the already-running β·0.8% forked2_qk_aon is skipped once
# its FINAL lands). Distinct _qk_aon run-names. --subtract-contrastive-floor + GRU-ckpt are
# inherited from train_backbone_b1024_1gpu.sh.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RES="$OUT/results"; mkdir -p "$RES"; OLOG="$RES/orchestrate_qkaon.log"
STEPS="${STEPS:-12500}"; SAVE_EVERY="${SAVE_EVERY:-2500}"; ALLT_CHUNK="${ALLT_CHUNK:-2}"
GPU="${GPU:-1}"; GATE_MIB="${GATE_MIB:-22000}"
export QK_NORM=1 ATTN_OUT_NORM=1 LR="${LR:-1e-3}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$(($1+1))p"; }
wait_for_gpu(){ local w=0; until [ "$(free_mib "$GPU")" -ge "$GATE_MIB" ]; do
  [ $((w%600)) -eq 0 ] && log "WAIT gpu$GPU free=$(free_mib "$GPU") MiB (need $GATE_MIB)"; sleep 120; w=$((w+120)); done; }

#         arm      mix         tag                 chunk
ARMS=(
  "beta     0.0078125   forked2_qk_aon       2"
  "beta     0.10        forked10pct_qk_aon   2"
  "alltime  0.5         forked_qk_aon        $ALLT_CHUNK"
  "alltime  0.10        forked10pct_qk_aon   $ALLT_CHUNK"
  "alltime  0.0078125   forked2_qk_aon       $ALLT_CHUNK"
)
log "ORCHESTRATOR(qk+aon) start: GPU$GPU steps=$STEPS qk_norm+attn_out_norm lr=$LR"
for spec in "${ARMS[@]}"; do
  set -- $spec; arm="$1"; mix="$2"; tag="$3"; chunk="$4"
  log ">>> ARM $arm $tag (mix=$mix)"
  wait_for_gpu
  bash "$HERE/train_backbone_b1024_1gpu.sh" "$arm" "$mix" "$tag" "$STEPS" "$SAVE_EVERY" "$chunk" "$GPU"
  log "<<< ARM $arm $tag rc=$?"
done
log "ORCHESTRATOR(qk+aon) DONE"; touch "$RES/.qkaon_backbones_done"
