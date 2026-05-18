#!/bin/bash
# v9c — resume v7 backbone (heads+layers-shared dropkey=0.9, fp32) from
# its best_loss checkpoint (step 50000, ema_loss 1.337) and train to
# step 100000. v7 was still descending slightly at 50k (last 5k drop
# ~0.4%); test if the backbone has more juice at the same recipe.
#
# Decision tree if MASE drops vs v7's 1.512:
#   - v9c MASE < v7 MASE: more backbone training is the lever → try
#     even longer (166k = 1 epoch).
#   - v9c MASE >= v7 MASE: backbone hit its ceiling → focus on q-head
#     architecture / forecast strategy / loss-shape.
#
# GPU 1 (parallel with v9a q-head 60k on GPU 0).

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_v7_continue_100k"
RESUME="/home/jupyter/contrastive-forecasting/checkpoints/enc_fcst_dk09_hsl_b256_fp32_50k_best_loss.pth"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

[ -f "$RESUME" ] || { echo "ERROR: v7 best_loss missing at $RESUME" >&2; exit 1; }
[ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }

echo "=== START ${NAME} (resume v7 best_loss → 100k) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 100000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
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
