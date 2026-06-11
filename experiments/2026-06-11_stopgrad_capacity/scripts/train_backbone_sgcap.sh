#!/bin/bash
# #341 — stop-grad × capacity: two new backbones, both with --stopgrad-positive-h
# (SimSiam/BYOL-style stop-grad on the encoder side h_{t+1} of the InfoNCE
# positive, numerator and denominator), differing only in capacity allocation:
#   nobn_enc6  6-layer encoder + full-width (384/6) 6-layer forecaster   [arm 3]
#   bn_enc6    6-layer encoder + 128-wide/4-head bottleneck forecaster   [arm 4]
# Everything else is the EXACT #339 recipe (allt·0.8% + crossfade triplet,
# qk-norm, attn-out-norm, --subtract-contrastive-floor, tau 0.10, batch 1024,
# 12 500 steps, seed 20260520). Single GPU per arm.
#
#   train_backbone_sgcap.sh <arm: nobn_enc6|bn_enc6> <gpu> [steps] [save_every]
set -uo pipefail
ARM="${1:?arm (nobn_enc6|bn_enc6)}"; GPU="${2:?gpu}"; STEPS="${3:-12500}"; SAVE_EVERY="${4:-2500}"
SEED=20260520
WT="${WT:-/tmp/cf-341}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity}"
case "$ARM" in
  nobn_enc6) EXTRA=() ;;                                          # full-width forecaster
  bn_enc6)   EXTRA=(--forecaster-d-model 128 --forecaster-n-heads 4) ;;  # v13 bottleneck
  *) echo "unknown arm: $ARM (want nobn_enc6|bn_enc6)"; exit 2 ;;
esac
NAME="bb_allt08_xftrip_${ARM}_sgpos_qk_aon_b1024"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-2}" CUDA_VISIBLE_DEVICES="$GPU"
# GRU patch-encoder memory management for 1-GPU batch 1024 (byte-identical;
# chunks + gradient-checkpoints the GRU). Same settings as #328/#339 backbones.
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-sgcap-$ARM g$GPU] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
# Crash resilience: resume from the latest periodic checkpoint (model+opt+step+RNG+data).
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START arm=$ARM bs=1024 steps=$STEPS chunk=$XSHH_ALLT_CHUNK ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 "${EXTRA[@]}" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --stopgrad-positive-h \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
