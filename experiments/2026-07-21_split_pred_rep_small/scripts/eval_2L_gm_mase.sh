#!/bin/bash
# 2L quantile-head + GIFT-Eval B4 on a #379 arm's 40k backbone.
# Usage: ARM=<slug> BB_GPU=<0|1> [BB_STEP_K=40] [HEAD_STEPS=15000] \
#          bash eval_2L_gm_mase.sh
#
# Trains a fresh 2L transformer quantile head (head_nhead=8 to divide
# d_model=64) on the frozen backbone at step ${BB_STEP_K}k for ${HEAD_STEPS}
# steps, then runs GIFT-Eval B4 (full 97 configs, forecast_len=16).
# Aggregate GM-Relative MASE is on the first line of summary.txt.
set -uo pipefail

WT=/home/jupyter/wt-cf-379-train
EXP="$WT/experiments/2026-07-21_split_pred_rep_small"
RUNS="$EXP/runs"
SCRIPTS="$EXP/scripts"
OUT_ROOT="$EXP/eval_gm_mase"
HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
GEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"

export PYTHONPATH="$WT"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"

ARM="${ARM:?set ARM=<slug>}"
BB_GPU="${BB_GPU:?set BB_GPU=0 or 1}"
BB_STEP_K="${BB_STEP_K:-40}"
HEAD_STEPS="${HEAD_STEPS:-15000}"

# Find the arm's backbone base run-name via run_arm.sh's NAME= assignment.
NAME=$(awk -v pat="^[[:space:]]*${ARM})" 'BEGIN{on=0} $0 ~ pat {on=1; next} on && /NAME=/{print; on=0}' "$SCRIPTS/run_arm.sh" | grep -oE 'NAME="[^"]+"' | head -1 | sed 's/NAME="//;s/"$//')
[ -n "$NAME" ] || { echo "ABORT: could not resolve NAME for arm '$ARM'" >&2; exit 2; }

# Locate the ${BB_STEP_K}k.pth checkpoint. Each resume appends a fresh
# `_r<N>` safe-run-name suffix, so match any of them, newest first.
BB=$(ls -t "$RUNS/${NAME}"_${BB_STEP_K}k.pth "$RUNS/${NAME}"_r*_${BB_STEP_K}k.pth 2>/dev/null | head -1)
[ -f "$BB" ] || { echo "ABORT: no ${BB_STEP_K}k backbone for arm '$ARM' (name=$NAME)" >&2; exit 3; }

CELL="${ARM}_bb${BB_STEP_K}k_hd${HEAD_STEPS}s"
OUT="$OUT_ROOT/$CELL"; mkdir -p "$OUT"
HEAD_NAME="qhead_2L_${NAME}_bb${BB_STEP_K}k"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/eval.log"

echo "[$(date +%H:%M:%S)] $ARM cell=$CELL start on GPU $BB_GPU (backbone=$(basename $BB))" | tee -a "$LOG"

# Backbone arch args accepted by train_forecasting_head.py (the head trainer
# loads the extra backbone flags from the checkpoint automatically).
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)
# GIFT-Eval doesn't accept freq/seasonality embedding flags (they're
# reconstructed from checkpoint config); everything else it does.
ARCH_EVAL=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

# --- 1) Train 2L transformer quantile head ---------------------------------
if [ ! -f "$HEAD_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] $ARM head-train ${HEAD_STEPS} steps" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$HEAD_TRAIN" \
    --backbone-path "$BB" \
    --device cuda \
    --quantile-head --grad-clip 1.0 \
    --forecast-len 16 --batch-size 256 --lr 1e-3 \
    --total-steps "$HEAD_STEPS" --save-every 5000 --log-every 500 \
    --save-dir "$OUT" --run-name "$HEAD_NAME" --seed 20260722 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --head-arch transformer --head-num-layers 2 --head-nhead 8 \
    --head-ffn-mult 4.0 --head-causal true --head-train-input e_then_f \
    --head-dropout 0.1 \
    "${ARCH_HEAD[@]}" >>"$LOG" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $ARM head-train rc=$rc" | tee -a "$LOG"
  [ $rc -eq 0 ] || exit $rc
else
  echo "[$(date +%H:%M:%S)] $ARM head-train SKIP (FINAL exists)" | tee -a "$LOG"
fi

# --- 2) GIFT-Eval B4 full-97 ---------------------------------------------
echo "[$(date +%H:%M:%S)] $ARM gift-eval start" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$GEVAL" \
  --backbone-path "$BB" \
  --head-path "$HEAD_CKPT" \
  --output-dir "$OUT/gift" \
  --strategy B4 --forecast-len 16 --resume \
  "${ARCH_EVAL[@]}" \
  --head-nhead 8 --head-causal true \
  >>"$LOG" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] $ARM gift-eval rc=$rc" | tee -a "$LOG"

# Aggregate result: line beginning with "Aggregate" in the gift subdir.
AGG=$(grep -h "Aggregate" "$OUT/gift"/*.txt "$OUT/gift"/**/*.txt 2>/dev/null | head -1)
if [ -n "$AGG" ]; then
  echo "$AGG" > "$OUT/summary.txt"
  echo "[$(date +%H:%M:%S)] $ARM DONE — $AGG" | tee -a "$LOG"
else
  echo "[$(date +%H:%M:%S)] $ARM WARN — no Aggregate line found" | tee -a "$LOG"
fi
