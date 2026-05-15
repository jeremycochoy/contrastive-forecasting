#!/bin/bash
# v14 — branched from v11c (fresh-init JEPA, NEW conv placement, PURE fp32).
# Only difference from v11c: forecaster = 6L (--num-layers 6) instead of 1L.
# Encoder still 6L, dk=0.9 share-heads/layers, depthwise-conv 3, deprecated 0.
#
# Hypothesis: v11c (1L forecaster) plateaus contrastive loss at ~2.10 but still
# yields a usable backbone (qhead 0.221, GM-MASE 1.388). A wider forecaster
# (6L) gives the model more capacity to fit the JEPA target — should lower
# the contrastive loss meaningfully; question is whether the encoder
# representations improve as a result (qhead ema_loss + GM-MASE).
#
# Compare:
#   v10  (1L fcst, legacy conv, fp32):  loss ~1.45 → qhead 0.260 → GM 1.437
#   v11c (1L fcst, new conv, fp32):     loss ~2.10 → qhead 0.221 → GM 1.388
#   v14  (6L fcst, new conv, fp32):     ??? → ??? → ???

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v14_jepa_enc6_fcst6_dk09_newconv_fp32_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (fresh, JEPA 6L+6L, new conv, PURE fp32) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --patch-emb-dtype fp32 --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} ===" && date
