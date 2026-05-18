#!/bin/bash
# v10d — control for v10c. Same resume from v6's _best_loss.pth (step
# 3800) but in pure fp32. Tests whether v6's weights themselves are
# "broken" or whether bf16+fp32loss is the cause of v10c's failure.
#
# Identical to v10c except --amp-dtype none.
# GPU 0. Run name: enc_fcst_v10d_fp32_resume_v6_15k.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v10d_fp32_resume_v6_15k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dk09_hsl_b256_50k_best_loss.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "$RESUME" ] || { echo "ERROR: v6 best_loss missing at $RESUME" >&2; exit 1; }
[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }

echo "=== START ${NAME} (resume v6 best_loss → step 15000, pure fp32) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 15000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 2000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
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
echo "=== DONE ${NAME} ===" && date
