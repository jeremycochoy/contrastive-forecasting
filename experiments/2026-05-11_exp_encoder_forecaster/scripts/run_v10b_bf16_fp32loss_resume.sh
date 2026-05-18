#!/bin/bash
# v10b — validate the new bf16-forward + fp32-loss path by resuming
# v10's checkpoint at step 5000 and continuing in bf16. Run side-by-
# side with v10 (still in pure fp32) for direct trajectory comparison.
#
# If v10b's loss/R²/U_b curves agree with v10's beyond the resume
# point, bf16+fp32loss is validated as a 50%-faster equivalent of
# pure fp32 for our high-dropkey + small-τ regime.
#
# Same recipe as v10:
#   --num-encoder-layers 6 --num-layers 1 (JEPA)
#   --encoder-dropkey 0.9 --encoder-dropkey-share-heads --encoder-dropkey-share-layers
#   B=256, lr=1e-3, τ=0.10
# CHANGE:
#   --amp-dtype bf16 --amp-keep-loss-fp32 true  (NEW path)
#   --resume v10_5k.pth                         (start from v10's step 5000)
#   --total-steps 25000                          (20k more steps for side-by-side)
#
# GPU 0 (parallel with v10 still on GPU 1).
# Run name: enc_fcst_v10b_bf16_fp32loss_resume5k_25k.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v10b_bf16_fp32loss_resume5k_25k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_v10_jepa_enc6_fcst1_50k_5k.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "$RESUME" ] || { echo "ERROR: v10 5k checkpoint missing at $RESUME" >&2; exit 1; }
[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }

echo "=== START ${NAME} (resume v10 step 5000 → step 25000, bf16+fp32loss) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 25000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --num-encoder-layers 6 --encoder-dropkey 0.9 \
    --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --amp-dtype bf16 --amp-keep-loss-fp32 true \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} ===" && date
