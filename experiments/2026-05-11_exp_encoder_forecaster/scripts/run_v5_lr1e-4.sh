#!/bin/bash
# v5 — lower LR (1e-4 vs 1e-3 in v4) to fix the v4 divergence at step ~4200.
#
# v4 (B=512, dropkey=0.9 heads-shared, lr=1e-3, fresh) showed the same
# single-step kick-out divergence pattern as v2/v3 — loss jumped 3.06 → 7.94
# at step 4200, then oscillated around 5–10 for the remaining ~1.5k steps.
# User call: drop LR by 10× rather than touching tau or dropkey strength
# again.
#
# All other config IDENTICAL to v4 (the run_b512_dk09_hs.sh recipe):
#   --batch-size 512
#   --encoder-dropkey 0.9 --encoder-dropkey-share-heads
#   --amp-dtype bf16
#   --tau 0.10
#   fresh from step 0 (no --resume)
#
# CHANGE: --lr 1e-4 (was 1e-3).
#
# Run name: enc_fcst_dk09_hs_b512_lr1e-4_50k

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dk09_hs_b512_lr1e-4_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "=== START ${NAME} (steps=${TOTAL_STEPS}, B=${BATCH_SIZE}, lr=1e-4) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
    --lr 1e-4 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 --encoder-dropkey-share-heads \
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
