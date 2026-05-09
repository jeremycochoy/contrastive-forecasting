#!/bin/bash
# Apples-to-apples learnable-τ baseline for the τ-sweep.
#
# Same recipe as run_tau_sweep_elisa.sh (fresh, 15k steps, batch=256, all
# other flags identical) but with --learnable-tau --tau 0.10. The τ-sweep
# arms are fixed-τ from scratch; backbone-beta_167k is learnable-τ but
# resumed-from-50k and 11× longer training. This run lets us compare
# fixed-τ vs learnable-τ at the same recipe.

set -e
cd /home/jupyter/contrastive-forecasting

export PYTHONPATH=/home/jupyter/contrastive-forecasting
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"
SAVE_DIR="sync_tau_sweep_learnable/checkpoints"
NAME="tau_sweep_learnable_0_10"

mkdir -p "${SAVE_DIR}"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ===" && date && exit 0
fi

echo "" && echo "=== ARM learnable τ init=0.10 → run_name=${NAME} ===" && date

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
    --tau 0.10 --learnable-tau \
    --loss-shape "${LOSS}"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
echo "=== ARM learnable τ DONE — saved ${NAME}_FINAL.pth ===" && date
exit 0
