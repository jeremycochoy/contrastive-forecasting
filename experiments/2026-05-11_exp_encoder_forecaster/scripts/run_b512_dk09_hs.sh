#!/bin/bash
# Fresh-from-zero attempt with the multi-factor change agreed at ~03:00:
#   --batch-size 512                  (was 256; halves effective LR per sample,
#                                      reduces outlier-batch impact)
#   --encoder-dropkey 0.9             (was 0.7; stronger past-key drop, also
#                                      higher per-step noise)
#   --encoder-dropkey-share-heads     (heads see same mask within a (B, layer);
#                                      prevents heads cooperating to count
#                                      positions)
#   --lr 1e-3                         (UNCHANGED — relying on the bigger batch
#                                      to do the LR work)
#   FRESH FROM STEP 0                 (no --resume; clean trajectory for the
#                                      curve)
#
# Run name: enc_fcst_dk09_hs_b512_50k
#
# Memory note: B=256 + bf16 used ~11 GB on the 24 GB 4090. B=512 should
# land around ~22 GB — right at the edge. If it OOMs in the first ~100
# steps, fall back to B=384.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dk09_hs_b512_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "=== START ${NAME} (steps=${TOTAL_STEPS}, B=${BATCH_SIZE}) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size "${BATCH_SIZE}" \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
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
