#!/bin/bash
# v6 — dropkey mask shared across heads AND across all 6 encoder layers.
#
# Hypothesis: with per-layer independent draws at p=0.9, P(token blocked
# in *every* layer) = p^L = 0.9^6 ≈ 53%. The other 47% of tokens are
# visible at *some* layer — enough position information for an implicit
# counter to recover via the union-across-layers signal. Layer-shared
# mask → P(blocked) = p = 90% flat across every layer for every token,
# so 90% of positions are completely invisible to attention everywhere
# in the encoder stack. Much harder to count.
#
# Otherwise identical to v3-era baseline:
#   --batch-size 256 (back to original from v4/v5's 512)
#   --lr 1e-3        (back to original from v5's 1e-4)
#   --encoder-dropkey 0.9
#   --encoder-dropkey-share-heads
#   --encoder-dropkey-share-layers  ← NEW
#   --amp-dtype bf16
#   --tau 0.10
#   fresh from step 0 (no --resume)
#
# Run name: enc_fcst_dk09_hsl_b256_50k (hsl = heads+layers-shared).

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dk09_hsl_b256_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"

echo "=== START ${NAME} (steps=${TOTAL_STEPS}, B=${BATCH_SIZE}, lr=1e-3) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --amp-dtype bf16 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} — saved ${NAME}_FINAL.pth ===" && date
