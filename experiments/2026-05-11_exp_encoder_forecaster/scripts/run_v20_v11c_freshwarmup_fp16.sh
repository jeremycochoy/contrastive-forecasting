#!/bin/bash
# v20 — two-phase "fp32 warmup → fp16" validation of the v11c recipe with a
# FRESH seed (≠ v11c's original init). Serves two goals:
#   (1) speedup recipe: if final MASE ≈ v11c's 1.29, fp32-warmup-then-fp16 is
#       a valid faster recipe (only ~5k of the 50k steps are slow fp32).
#   (2) lucky-init check: fresh seed; if MASE ≈ 1.29, v11c was not a lucky init.
#
# Phase A: fresh fp32 v11c-recipe, 0 → 5000 steps (the stable warmup).
# Phase B: resume Phase-A _5k.pth with fp16 body (resid+attn+ffn fp16,
#          patch-emb fp32), continue 5000 → 50000.
# v18 proved this recipe diverges WITHOUT the warmup at fresh init.

set -euo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

SAVE_DIR="$MAIN/checkpoints"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"
A_NAME="enc_fcst_v20_phaseA_fp32warmup_5k"
B_NAME="enc_fcst_v20_v11c_freshwarmup_fp16_50k"

COMMON_ARGS=(
    --device cuda --batch-size 256
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --save-dir "$SAVE_DIR"
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1"
    --t-raw 4096 --n-channels 1
    --d-model 384 --n-heads 6 --num-layers 1
    --num-encoder-layers 6 --encoder-dropkey 0.9
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers
    --depthwise-conv 3 --deprecated-depthwise-conv 0
    --mix-ratio 0.0
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3
    --rev-norm-kind ewma --rev-norm-span 128
    --tau 0.10
    --loss-shape "cosine_similarity_batch"
    --encoder-type gru
)

# ---- Phase A: fresh fp32 warmup 0 → 5000 ----
if [ ! -f "$SAVE_DIR/${A_NAME}_5k.pth" ]; then
    echo "=== v20 Phase A START (fresh fp32 warmup → 5k) ===" && date
    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        "${COMMON_ARGS[@]}" \
        --total-steps 5000 --save-every 5000 --run-name "$A_NAME" \
        --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
        2>&1 | tee -a "$LOG_DIR/run_${A_NAME}.log"
fi
[ -f "$SAVE_DIR/${A_NAME}_5k.pth" ] || { echo "ERROR: Phase A _5k.pth missing" >&2; exit 1; }

# ---- Phase B: resume _5k with fp16 body → 50000 ----
if [ ! -f "$SAVE_DIR/${B_NAME}_FINAL.pth" ]; then
    echo "=== v20 Phase B START (resume _5k, fp16 body → 50k) ===" && date
    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        "${COMMON_ARGS[@]}" \
        --total-steps 50000 --save-every 5000 --run-name "$B_NAME" \
        --resume "$SAVE_DIR/${A_NAME}_5k.pth" \
        --patch-emb-dtype fp32 --residual-dtype fp16 --attn-dtype fp16 --ffn-dtype fp16 \
        2>&1 | tee -a "$LOG_DIR/run_${B_NAME}.log"
    cp -f "$SAVE_DIR/${B_NAME}_best_loss.pth" "$SAVE_DIR/${B_NAME}_FINAL.pth"
fi
echo "=== v20 DONE (final = $SAVE_DIR/${B_NAME}_FINAL.pth) ===" && date
