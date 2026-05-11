#!/bin/bash
# Resume from enc_fcst_dropkey07_pb_50k_best_loss.pth with HEADS-SHARED
# dropkey. Triggered by attempt-2 (per-(B,head) p=0.7) divergence at
# step ~14900: ema_loss climbed 2.07 → 5.21 → 7+ over 1000 steps with
# no recovery. Per-(B,head) variance at p=0.7 was too high.
#
# Heads-shared dropkey: independent per (batch_row, layer), but ALL
# heads within a (batch_row, layer) see the IDENTICAL mask. Variance
# drops by ~num_heads× (=6×) and forces heads to disagree on which
# positions to attend to (rather than cooperating to count).
#
# Run name: enc_fcst_dropkey07_pb_hs_50k (kept "_pb_" prefix because
# batch + layer are still per-(B,layer) independent, just heads tied).
# --resume from the 50k attempt-2's _best_loss (the lowest-loss
# checkpoint before divergence, ~step 10500).

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dropkey07_pb_hs_50k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dropkey07_pb_50k_best_loss.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ ! -f "$RESUME" ]; then
    echo "ERROR: resume checkpoint missing at $RESUME" >&2
    exit 1
fi
if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (resume from attempt-2 best_loss → ${TOTAL_STEPS}) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.7 --encoder-dropkey-share-heads \
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
