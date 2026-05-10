#!/bin/bash
# Transformer-encoder run #1 — replaces the GRU+skip patch encoder with a
# 4-layer decoder-only causal transformer (Linear(W'->H) per-patch, then
# 4 layers attending over T patches). Architecture / training recipe is
# otherwise identical to the τ=0.10 baseline at experiments/2026-05-08_
# exp_tau_sweep/RESULTS.md so we have an apples-to-apples comparison.
#
# Reference baseline (sweep table, N=50 held-out):
#   tau_sweep_0_10  (GRU+skip)  AUC=0.8993 ± 0.0053  Top1=0.7535 ± 0.0098
#                                R²_random=0.6683    R²_naive=0.6153
#                                U_t=0.0512   U_b=0.1019
#
# Code lives on branch transformer-encoder-experiment (worktree at
# /home/jupyter/cf-transformer-encoder/). Save dir is in the MAIN checkout
# at /home/jupyter/contrastive-forecasting/sync_transformer_encoder/ so
# `git worktree remove --force` on the worktree can never delete checkpoints
# (CLAUDE.md rule 4).

set -e

WORKTREE=/home/jupyter/cf-transformer-encoder
MAIN_CHECKOUT=/home/jupyter/contrastive-forecasting

cd "${WORKTREE}"

export PYTHONPATH="${WORKTREE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"
SAVE_DIR="${MAIN_CHECKOUT}/sync_transformer_encoder/checkpoints"
NAME="transformer_encoder_tau_0_10_50k"
LOG_FILE="${MAIN_CHECKOUT}/sync_transformer_encoder/run.log"

mkdir -p "${SAVE_DIR}" "$(dirname "${LOG_FILE}")"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth already exists ==="
    date
    exit 0
fi

echo "" && echo "=== TRANSFORMER-ENCODER τ=0.10 (50k from scratch) — ${NAME} ==="
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
date

python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 50000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --encoder-type transformer \
    --enc-num-layers 4 --enc-nhead 6 --enc-ffn-mult 4.0 \
    --enc-dropout 0.0 --enc-depthwise-conv 3 \
    --tau 0.10 \
    --loss-shape "${LOSS}"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
echo "=== TRANSFORMER-ENCODER 50k DONE — saved ${NAME}_FINAL.pth ===" && date
exit 0
