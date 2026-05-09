#!/bin/bash
# Continue τ=0.20 from its 15k v2 FINAL.pth to 50,000 steps total, so we
# can compare its long-window trajectory against backbone-β_167k.
#
# Resume bundle (model + optimizer + step counter + RNG state):
#   sync_tau_sweep_arm5_v2/checkpoints/tau_sweep_0_20_v2_FINAL.pth

set -e
cd /home/jupyter/contrastive-forecasting

export PYTHONPATH=/home/jupyter/contrastive-forecasting
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"
SAVE_DIR="sync_tau_sweep/checkpoints"
NAME="tau_sweep_0_20_50k"
RESUME="sync_tau_sweep_arm5_v2/checkpoints/tau_sweep_0_20_v2_FINAL.pth"

mkdir -p "${SAVE_DIR}"

if [ ! -f "${RESUME}" ]; then
    echo "FATAL: resume file ${RESUME} missing"; exit 1
fi
if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ===" && date && exit 0
fi

echo "" && echo "=== ARM τ=0.20 RESUME → 50k → run_name=${NAME} ===" && date

python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 50000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
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
echo "=== ARM τ=0.20 50k DONE — saved ${NAME}_FINAL.pth ===" && date
exit 0
