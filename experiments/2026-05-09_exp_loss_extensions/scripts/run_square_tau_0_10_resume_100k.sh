#!/bin/bash
# Continue loss_ext_square_tau_0_10 from step 15000 to step 100000.
# Resume bundle: sync_loss_ext_square/checkpoints/loss_ext_square_tau_0_10_15k.pth
# (companion _optimizer.pth is loaded automatically from same dir)
#
# This is the long-trajectory variant. Used to verify whether the
# square loss continues to track / cross the baseline at late steps,
# now that baselines have a clean 50k plateau and 100k will let us
# look at whether square reaches the same plateau or saturates lower.

set -e
cd /home/jupyter/contrastive-forecasting

export PYTHONPATH=/home/jupyter/contrastive-forecasting
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

SAVE_DIR="sync_loss_ext_square/checkpoints"
NAME="loss_ext_square_tau_0_10_100k"
RESUME="${SAVE_DIR}/loss_ext_square_tau_0_10_15k.pth"

if [ ! -f "${RESUME}" ]; then
    echo "FATAL: resume file ${RESUME} missing"; exit 1
fi
if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="; date; exit 0
fi

echo "=== loss_ext_square τ=0.10 RESUME → 100k → ${NAME} ==="; date

python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 100000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --encoder-type gru \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch_square"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
echo "=== ${NAME} DONE ==="; date
