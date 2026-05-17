#!/bin/bash
# v10 — JEPA-style: heavy encoder (6L), tiny forecaster (1L).
#
# Hypothesis: in v7, R² on latent retrieval is ~0.998 but downstream
# MASE is 1.512. The 6-layer forecaster may be doing the prediction
# work — the encoder is lazy, just produces enough signal for the
# 6L forecaster to extract during contrastive training. Then the
# downstream q-head, which sees only the encoder output, can't
# reconstruct what the forecaster was doing internally.
#
# Fix per JEPA: cripple the forecaster (1L). The contrastive loss
# can only descend if the encoder PRODUCES rich enough features that
# a 1L predictor suffices. Forces all semantic work into the encoder (will be paired with a 2L qhead, JEPA-style)
# = the part the q-head sees.
#
# Identical to v7 otherwise:
#   --num-encoder-layers 6  (UNCHANGED)
#   --num-layers 1          (was 6; drop forecaster to 1 layer)
#   --encoder-dropkey 0.9 --encoder-dropkey-share-heads --encoder-dropkey-share-layers
#   B=256, lr=1e-3, fp32, τ=0.10, fresh from 0, 50k steps
#
# Param count: ~14M (vs v7's 22M) — most savings from dropping
# 5 forecaster layers.
#
# Run name: enc_fcst_v10_jepa_enc6_fcst1_50k.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v10_jepa_enc6_fcst1_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (encoder=6L, forecaster=1L, ${TOTAL_STEPS} steps) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
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
echo "=== DONE ${NAME} ===" && date
