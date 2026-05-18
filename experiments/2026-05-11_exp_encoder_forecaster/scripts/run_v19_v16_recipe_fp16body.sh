#!/bin/bash
# v16 — branched from v11c (fresh-init JEPA, NEW conv placement, PURE fp32).
# Only difference from v11c: --encoder-dropkey 0.7 (was 0.9 in v11c).
# Forecaster still 1L, encoder still 6L, dk share-heads/layers, depthwise-conv 3, deprecated 0.
#
# Hypothesis: Wave-1 (v14 1L→6L fcst, v15 1L→4L fcst) regressed vs v11c, confirming the
# 1L-forecaster JEPA story. Wave-2 keeps forecaster=1L but loosens dropkey from 0.9 → 0.7
# to test whether the encoder is over-regularized at dk=0.9.
#
# Compare:
#   v11c (1L fcst, dk=0.9):  qhead 0.221 → GM 1.388
#   v14  (6L fcst, dk=0.9):  qhead ?     → GM 1.650 (regressed)
#   v15  (4L fcst, dk=0.9):  qhead ?     → GM 1.671 (regressed)
#   v16  (1L fcst, dk=0.7):  ??? → ??? → ???

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v19_v16_recipe_fp16body_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (fresh, JEPA 6L+1L, dk=0.7, new conv, PURE fp32) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --num-encoder-layers 6 --encoder-dropkey 0.7 \
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
