#!/bin/bash
# v7 — same as v6 but pure fp32 instead of bf16.
#
# Hypothesis: v4/v5/v6 all diverged at step counts that track LR (lr=1e-3
# → step ~4200, lr=1e-4 → step 30000) independent of mask sharing axis.
# This looks more like a numerical instability in the bf16 forward+loss
# than a dropkey property. Test: same recipe but --amp-dtype none (pure
# fp32). If v7 survives past where v6 diverged (~step 4500), bf16 was
# the culprit.
#
# Compute cost: fp32 roughly halves throughput vs bf16 on the 4090, but
# the user's instruction was explicit. Memory at B=256 + fp32 + 12L
# transformers + dropkey mask: estimated ~16-18 GB, well within 24 GB.
#
# Run name: enc_fcst_dk09_hsl_b256_fp32_50k.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dk09_hsl_b256_fp32_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

echo "=== START ${NAME} (steps=${TOTAL_STEPS}, B=${BATCH_SIZE}, lr=1e-3, fp32) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --amp-dtype none \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} — saved ${NAME}_FINAL.pth ===" && date
