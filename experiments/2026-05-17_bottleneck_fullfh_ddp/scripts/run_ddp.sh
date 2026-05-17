#!/bin/bash
# 2-GPU DDP backbone train — v13 forecaster-bottleneck (6L enc / 2L fcst
# d128-4h), dropkey 0.70 shared, AdamW wd0.1 β2=0.95, normalized-InfoNCE
# full (f_t,h_l)-negatives loss, residual fp32 + attn/ffn/conv bf16,
# global batch 256 (128/GPU) with full cross-rank negatives, 50k steps.
# Code from the conv-dtype worktree (PR #294); artifacts in the main
# checkout (survive worktree teardown). Idempotent: skips if FINAL exists.
set -uo pipefail

CODE=/home/jupyter/cf-wt-bottleneck-fullfh
EXP=/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp
TRAIN="$CODE/experiments/2026-04-27_freq-embedding/scripts/train.py"
SAVE_DIR="$EXP/runs"
RES="$EXP/results"
NAME="enc_fcst_bneck128_dk07_fullfh_norminfonce_ddp_50k"
LOG="$RES/run_${NAME}.log"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
SEED="${SEED:-20260517}"
mkdir -p "$SAVE_DIR" "$RES"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] || [ -f "${SAVE_DIR}/${NAME}_final.pth" ]; then
  echo "=== SKIP — ${NAME} already finished ===" | tee -a "$LOG"; exit 0
fi

cd "$CODE"
export PYTHONPATH="$CODE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
HF_TOKEN_FILE=/home/jupyter/contrastive-forecasting/experiments/hf_token.txt
export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Quiet, deterministic DDP; pick a free master port to avoid collisions
# with any concurrent torchrun on this shared box.
export OMP_NUM_THREADS=8
MASTER_PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"

echo "=== START ${NAME} | $(date) | port ${MASTER_PORT} | code @ $(git -C "$CODE" rev-parse --short HEAD) ===" | tee -a "$LOG"

torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" "$TRAIN" \
  --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 128 \
  --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
  --seed "${SEED}" \
  --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
  --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
  --t-raw 4096 --n-channels 1 \
  --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 2 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 \
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --residual-dtype fp32 --attn-dtype bf16 --ffn-dtype bf16 \
  --conv-dtype bf16 --patch-emb-dtype fp32 \
  --loss-shape cosine_similarity_batch_full_fh_negs --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 \
  --encoder-type gru --mixup-p 0.3 --mix-ratio 0.0 \
  --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

echo "=== train.py exit rc=${RC} | $(date) ===" | tee -a "$LOG"
if [ "$RC" -eq 0 ]; then
  # Checkpoint-safety rule #1: permanent FINAL copy off _best_loss/_final.
  if   [ -f "${SAVE_DIR}/${NAME}_best_loss.pth" ]; then cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
  elif [ -f "${SAVE_DIR}/${NAME}_final.pth" ];     then cp -f "${SAVE_DIR}/${NAME}_final.pth"     "${SAVE_DIR}/${NAME}_FINAL.pth"; fi
  echo "=== DONE ${NAME} | FINAL=${SAVE_DIR}/${NAME}_FINAL.pth ===" | tee -a "$LOG"
fi
exit "$RC"
