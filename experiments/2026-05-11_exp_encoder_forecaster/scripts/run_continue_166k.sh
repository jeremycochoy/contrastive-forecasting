#!/bin/bash
# Continuation: resume enc_fcst_dropkey07_pb_50k from best_loss → train to
# 166k steps (≈ one full epoch on the 42.5M-window gift-pretrain-full-4096
# small_v1 dataset, batch=256). User gave overnight runtime + free compute,
# so the upper bound is "one epoch", with plateau-driven early stop at the
# operator's discretion (we judge plateau on Q + loss slope after the
# trigger script reads the CSV).
#
# Run name: enc_fcst_dropkey07_pb_166k (NEW save-path per CLAUDE.md #2 —
# never reuse --save-path when resuming). The 50k backbone's _best_loss
# and _best_loss_optimizer remain untouched.
#
# Architecture/HP — IDENTICAL to the 50k run (per-(B,head) dropkey 0.7,
# bf16, all HPs match). Only `--total-steps`, `--resume`, and `--run-name`
# differ.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dropkey07_pb_166k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dropkey07_pb_50k_best_loss.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ ! -f "$RESUME" ]; then
    echo "ERROR: 50k best_loss missing at $RESUME" >&2
    exit 1
fi
if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-166000}"

echo "=== START ${NAME} (resume from 50k_best_loss → ${TOTAL_STEPS}) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.7 --amp-dtype bf16 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} — saved ${NAME}_FINAL.pth ===" && date
