#!/bin/bash
# #309 — BYOL alignment term (λ=1) on top of the #303 arm-B contrastive
# loss. Single-variable add vs the arm-B baseline (full_hh_negs, the best
# contrastive arm): the ONLY changes vs that recipe are the two new flags
#   --align-loss-weight 1.0      (λ·(2−2·cos(f_t, sg(h_{t+1}))), non-saturating)
#   --subtract-contrastive-floor (cosmetic re-base, gradient-neutral)
# 1L forecaster + residual fp32 / attn-ffn-conv bf16 = the stable fp16
# recipe (#296: the 2L/bf16 form diverged). Global batch 256 (128/GPU), 50k.
#
# SET FOR YOUR BOX: CODE (repo checkout) and EXP (this experiment dir).
# Artifacts land under the MAIN checkout so they survive worktree teardown.
set -uo pipefail

CODE="${CODE:?set CODE=/path/to/contrastive-forecasting checkout}"
EXP="${EXP:?set EXP=/path/to/experiments/2026-05-21_align-loss}"
TRAIN="$CODE/experiments/2026-04-27_freq-embedding/scripts/train.py"
SAVE_DIR="$EXP/runs"
RES="$EXP/results"
NAME="enc_fcst_bneck128_armB_align1_floor_ddp_50k"
LOG="$RES/run_${NAME}.log"
TOTAL_STEPS="${TOTAL_STEPS:-50000}"
SEED="${SEED:-20260521}"
mkdir -p "$SAVE_DIR" "$RES"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ] || [ -f "${SAVE_DIR}/${NAME}_final.pth" ]; then
  echo "=== SKIP — ${NAME} already finished ===" | tee -a "$LOG"; exit 0
fi

cd "$CODE"
export PYTHONPATH="$CODE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
HF_TOKEN_FILE="$CODE/experiments/hf_token.txt"
export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
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
  --num-encoder-layers 6 --num-layers 1 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 \
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --residual-dtype fp32 --attn-dtype bf16 --ffn-dtype bf16 \
  --conv-dtype bf16 --patch-emb-dtype fp32 \
  --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
  --align-loss-weight 1.0 --subtract-contrastive-floor \
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
