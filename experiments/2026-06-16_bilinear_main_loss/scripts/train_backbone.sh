#!/bin/bash
# #350 — learnable log-bilinear W in the MAIN contrastive loss, on the best
# no-encoder + CPC arm from #348. One arm changes vs that baseline:
#   bilinear  main loss exp(uᵀv/τ) → exp(uᵀ W v), τ dropped, W learnable
#             (--main-loss-bilinear), W init (1/τ₀)·I, excluded from weight decay.
# `cpc` reproduces #348's +CPC τ=0.10 baseline (no change) for an optional
# same-machine cross-check; the canonical baseline is #348's saved cpc results.
# Everything else is byte-for-byte the #348 +CPC no-encoder recipe (GRU
# patch-embedding, d_model 384 / 6 heads, 6-layer forecaster, allt·0.8% +
# crossfade triplet, qk-norm, attn-out-norm, xshh_allt loss,
# --pos-in-denominator, --subtract-contrastive-floor, --stopgrad-positive-h,
# CPC --cpc-infonce-weight 1.0, batch 1024, 12 500 steps, seed 20260520).
# XSHH_ALLT_CHUNK=16 (default 1 in #348) is byte-identical to the loss
# (memory↔kernel launches) — only faster.
#   train_backbone.sh <arm: cpc|bilinear> <gpu> [steps] [save_every]
set -uo pipefail
ARM="${1:?arm (cpc|bilinear)}"; GPU="${2:?gpu}"; STEPS="${3:-12500}"; SAVE_EVERY="${4:-2500}"
SEED=20260520
WT="${WT:-/home/jupyter/cf-wt-350-bilinear}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss}"
case "$ARM" in
  cpc)      EXTRA=(--cpc-infonce-weight 1.0) ;;
  bilinear) EXTRA=(--cpc-infonce-weight 1.0 --main-loss-bilinear --main-bilinear-init-tau 0.10) ;;
  *) echo "unknown arm: $ARM (want cpc|bilinear)"; exit 2 ;;
esac
NAME="bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_${ARM}"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
export FCST_GRAD_CKPT="${FCST_GRAD_CKPT:-1}" XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-16}"
export CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT="${PATCH_ENC_CKPT:-1}" PATCH_ENC_CHUNK="${PATCH_ENC_CHUNK:-4}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-$ARM g$GPU] $*"; }
[ -n "$HF_TOKEN" ] || { log "WARN: empty HF_TOKEN — HF stream will throttle the GPU"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START arm=$ARM bs=1024 steps=$STEPS xshh_chunk=$XSHH_ALLT_CHUNK ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 0 --num-layers 6 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --stopgrad-positive-h "${EXTRA[@]}" \
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
