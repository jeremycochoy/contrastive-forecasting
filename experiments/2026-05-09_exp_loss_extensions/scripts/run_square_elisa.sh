#!/usr/bin/env bash
# Exp: loss_extensions — cosine_similarity_batch_square, tau=0.10 fixed
# Same architecture/dataset/hparams as tau_sweep_0_10 (exp 2026-05-08).
# Run on elisa GPU 1. Produces run_name: loss_ext_square_tau_0_10
set -euo pipefail

WORKDIR=~/workspaces/contrastive-forecasting
RUN_NAME=loss_ext_square_tau_0_10
SAVE_DIR=${WORKDIR}/experiments/2026-05-09_exp_loss_extensions/results/${RUN_NAME}

cd "$WORKDIR"
mkdir -p "$SAVE_DIR"

export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

CUDA_VISIBLE_DEVICES=1 python train.py \
  --device cuda \
  --total-steps 50000 \
  --batch-size 256 \
  --lr 1e-3 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --save-every 5000 \
  --save-dir "$SAVE_DIR" \
  --run-name "$RUN_NAME" \
  --hf-repo "jeremycochoy/gift-pretrain-full-4096" \
  --hf-path "small_v1" \
  --t-raw 4096 \
  --n-channels 1 \
  --d-model 384 \
  --n-heads 6 \
  --num-layers 6 \
  --mix-ratio 0.0 \
  --freq-emb-dim 3 \
  --seasonality-emb-dim 3 \
  --mixup-p 0.3 \
  --rev-norm-kind ewma \
  --rev-norm-span 128 \
  --tau "0.10" \
  --loss-shape "cosine_similarity_batch_square" \
  2>&1 | tee "$SAVE_DIR/run_${RUN_NAME}.log"
