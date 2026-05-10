#!/bin/bash
# Continue the transformer-encoder run from its 50k FINAL.pth to 150k
# total steps, so we can compare its long-window trajectory against the
# τ=0.10 GRU baseline 150k run (`tau_sweep_0_10_150k`). Same recipe as
# the original run.sh — just `--resume` + `--total-steps 150000` + a
# fresh run-name to keep the new CSV / checkpoints separate from the
# 50k bundle.

set -e

WORKTREE=/home/jupyter/cf-transformer-encoder
MAIN_CHECKOUT=/home/jupyter/contrastive-forecasting

cd "${WORKTREE}"

export PYTHONPATH="${WORKTREE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"
SAVE_DIR="${MAIN_CHECKOUT}/sync_transformer_encoder/checkpoints"
NAME="transformer_encoder_tau_0_10_150k"
RESUME="${SAVE_DIR}/transformer_encoder_tau_0_10_50k_FINAL.pth"

mkdir -p "${SAVE_DIR}"

if [ ! -f "${RESUME}" ]; then
    echo "FATAL: resume file ${RESUME} missing"; exit 1
fi
if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    date
    exit 0
fi

echo "" && echo "=== TRANSFORMER-ENCODER RESUME → 150k → run_name=${NAME} ==="
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
date

python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 150000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --resume "${RESUME}" \
    --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --encoder-type transformer \
    --enc-num-layers 2 --enc-nhead 6 --enc-ffn-mult 4.0 \
    --enc-dropout 0.0 --enc-depthwise-conv 3 \
    --enc-chunk-size 16384 \
    --amp-dtype bf16 \
    --tau 0.10 \
    --loss-shape "${LOSS}"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
echo "=== TRANSFORMER-ENCODER 150k DONE — saved ${NAME}_FINAL.pth ===" && date
exit 0
