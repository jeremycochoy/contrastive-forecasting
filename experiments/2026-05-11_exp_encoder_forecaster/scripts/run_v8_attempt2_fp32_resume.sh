#!/bin/bash
# v8 — resume the FIRST diverging config (attempt-2) in fp32 instead of bf16.
#
# attempt-2 recipe: B=256, lr=1e-3, dropkey=0.7, per-(B,head) independent
# (NO share-heads, NO share-layers), bf16. Diverged at step ~14900.
# v8 takes attempt-2's _best_loss (step 10200, ema 1.32) and continues
# training with EVERY HP unchanged except --amp-dtype none.
#
# If v8 survives past step 14900 cleanly, that confirms bf16 was the
# cause of ALL the divergences in this series (not just v6/v4's
# step-4200 wall but also attempt-2's step-14900 wall).
#
# Runs on GPU 0 (v7 occupies GPU 1).
#
# Run name: enc_fcst_dk07_pb_fp32_resume_50k.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dk07_pb_fp32_resume_50k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dropkey07_pb_50k_best_loss.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ ! -f "$RESUME" ]; then
    echo "ERROR: attempt-2 best_loss missing at $RESUME" >&2; exit 1
fi
if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="; exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (resume attempt-2 best_loss → ${TOTAL_STEPS}, fp32, GPU 0) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.7 \
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
