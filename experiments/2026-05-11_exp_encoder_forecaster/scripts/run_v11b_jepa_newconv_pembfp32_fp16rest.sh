#!/bin/bash
# v11b — same as v11 but with --patch-emb-dtype fp32 (the user's fallback
# plan when v11 with all-fp16 diverged). Tests whether the patch embedding
# (RevEWMNorm + GRU) needs fp32 precision, with the rest of the
# transformer body still fp16.
#
# Recipe: fresh JEPA 6L+1L, new conv placement (--depthwise-conv 3,
# --deprecated-depthwise-conv 0), dropkey=0.9 hsl, B=256,
# lr=1e-3, τ=0.10, 50k steps. Loss + compute_metrics always fp32.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v11b_jepa_newconv_pembfp32_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (fresh, JEPA 6L+1L, new conv, pemb fp32 + rest fp16) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --patch-emb-dtype fp32 --residual-dtype fp16 --attn-dtype fp16 --ffn-dtype fp16 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} ===" && date
