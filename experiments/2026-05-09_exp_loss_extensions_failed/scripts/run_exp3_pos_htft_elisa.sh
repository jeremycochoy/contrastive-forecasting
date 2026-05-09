#!/bin/bash
# Exp 3: contrastive-loss extension — add (h_t, f_t) same-channel same-time
# as additional positive pair (multi-positive InfoNCE numerator).
# Architecture: WINNER from Exp 1+2 = gru @ τ=0.20, 15k steps fresh.
# Loss: cosine_similarity_batch_add_pos_htft (PR #179).
# Compare AUC/Top-1 trajectory vs the τ=0.20 baseline.

set -e
cd /home/jupyter/contrastive-forecasting

export PYTHONPATH=/home/jupyter/contrastive-forecasting
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch_add_pos_htft"
SAVE_DIR="sync_exp3_pos_htft/checkpoints"
NAME="exp3_pos_htft_tau_0_20"

mkdir -p "${SAVE_DIR}"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ===" && date && exit 0
fi

echo "" && echo "=== Exp 3 launching: gru @ τ=0.20 + (h_t,f_t) positive → ${NAME} ===" && date

python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 15000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --encoder-type gru \
    --tau 0.20 \
    --loss-shape "${LOSS}"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
echo "=== Exp 3 DONE — saved ${NAME}_FINAL.pth ===" && date
exit 0
