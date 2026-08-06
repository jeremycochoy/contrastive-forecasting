#!/bin/bash
# 2L quantile-head + GIFT-Eval B4 on a #379 arm's 40k backbone.
# Usage: ARM=<slug> BB_GPU=<0|1> [BB_STEP_K=40] [HEAD_STEPS=15000] \
#          [BB_CHECKPOINT=<path>] bash eval_2L_gm_mase.sh
#
# BB_CHECKPOINT names the backbone file directly. Needed only when a run was
# resumed and both `<name>_<K>k.pth` and `<name>_r<N>_<K>k.pth` exist: the
# resolver refuses to guess between them. It must be one of those two names —
# a file from another step or another run is refused rather than published
# under this cell's name.
#
# The cell carries what decides its number — the backbone replicate and the
# head seed — so two measurements never share a directory, a head or an
# aggregate:
#
#   arm5_bb40k_hd15000s            the base run's 40k backbone, wave seed
#   arm5_bb40k_r3_hd15000s         the same arm's third resume, same step
#   arm5_s20260723_bb40k_hd15000s  the same backbone, another head seed
#
# The base run's token and the wave seed's token are both empty, so every
# cell name already on disk is unchanged.
#
# Trains a fresh 2L transformer quantile head (head_nhead=8 to divide
# d_model=64) on the frozen backbone at step ${BB_STEP_K}k for ${HEAD_STEPS}
# steps, then runs GIFT-Eval B4 (full 97 configs, forecast_len=16).
# Aggregate GM-Relative MASE is on the first line of summary.txt.
#
# Exit codes 2-6 are `resolve_eval_checkpoint.sh`'s, propagated verbatim.
# This script's own setup abort is 20, so no number means two things. 25 is
# the bad-tag abort, E_BAD_TAG from the cell-identity library — the same
# number #390's `eval_arm.sh` uses for the same operator action.
set -uo pipefail

WT="${WT:-/home/jupyter/wt-cf-379-train}"
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

# The shared libraries. They come from this script's own checkout, not from
# $WT: a $WT that predates the resolver would otherwise fall back to the
# mtime pick without saying so. `cd -P`, because this file is reached through
# a `scripts/` symlink — the logical path walks back up to $WT and lands on
# exactly that stale checkout. Loaded here, before the knobs, because the
# cell-identity library holds the head-seed default one of them takes.
ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_RESOLVER="$ROOT/scripts/resolve_eval_checkpoint.sh"
CELL_IDENTITY="$ROOT/scripts/eval_cell_identity.sh"
[ -f "$CKPT_RESOLVER" ] || { echo "ABORT: no checkpoint resolver at $CKPT_RESOLVER" >&2; exit 20; }
[ -f "$CELL_IDENTITY" ] || { echo "ABORT: no cell-identity library at $CELL_IDENTITY" >&2; exit 20; }
# shellcheck source=/dev/null
source "$CELL_IDENTITY"

ARM="${ARM:?set ARM=<slug>}"
BB_GPU="${BB_GPU:?set BB_GPU=0 or 1}"
BB_STEP_K="${BB_STEP_K:-40}"
HEAD_STEPS="${HEAD_STEPS:-15000}"
# The head's init/data seed. One variable, so the number the head trains
# under and the number the cell is named after cannot disagree.
HEAD_SEED="${HEAD_SEED:-$EVAL_DEFAULT_HEAD_SEED}"
BB_CHECKPOINT="${BB_CHECKPOINT:-}"

# Find the arm's backbone base run-name via run_arm.sh's NAME= assignment.
NAME=$(awk -v pat="^[[:space:]]*${ARM})" 'BEGIN{on=0} $0 ~ pat {on=1; next} on && /NAME=/{print; on=0}' "$SCRIPTS/run_arm.sh" | grep -oE 'NAME="[^"]+"' | head -1 | sed 's/NAME="//;s/"$//')
[ -n "$NAME" ] || { echo "ABORT: could not resolve NAME for arm '$ARM'" >&2; exit 20; }

# Locate the ${BB_STEP_K}k.pth checkpoint. Each resume appends a fresh
# `_r<N>` safe-run-name suffix, so several files can match one step; the
# resolver aborts rather than pick between them, and prints what it chose.
BB=$(bash "$CKPT_RESOLVER" "$RUNS" "$NAME" "$BB_STEP_K" "$BB_CHECKPOINT") || exit $?

# The cell is named after the backbone it cites, replicate included, and the
# seed its head was trained under — or a `_r<N>` backbone, or a second seed,
# lands in the first one's directory and reuses its head and its aggregate.
REPL_TAG="$(replicate_tag "$NAME" "$BB_STEP_K" "$BB")" || {
  echo "ABORT: resolved checkpoint is not '$NAME' at ${BB_STEP_K}k: $BB" >&2
  exit $E_BAD_TAG; }

CELL="$(eval_cell_name "$ARM" "$BB_STEP_K" "$REPL_TAG" "$HEAD_STEPS" "$HEAD_SEED")" || exit $E_BAD_TAG
OUT="$OUT_ROOT/$CELL"; mkdir -p "$OUT"
HEAD_NAME="qhead_2L_${NAME}_bb${BB_STEP_K}k${REPL_TAG}"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/eval.log"

echo "[$(date +%H:%M:%S)] $ARM cell=$CELL start on GPU $BB_GPU (backbone=$BB)" | tee -a "$LOG"

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
    --save-dir "$OUT" --run-name "$HEAD_NAME" --seed "$HEAD_SEED" \
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
  # The number and the file it came from, together: `eval.log` is appended to
  # and is not what the analysis reads. The aggregate stays the first line.
  { echo "$AGG"; echo "backbone: $BB"; } > "$OUT/summary.txt"
  echo "[$(date +%H:%M:%S)] $ARM DONE — $AGG (backbone=$BB)" | tee -a "$LOG"
else
  echo "[$(date +%H:%M:%S)] $ARM WARN — no Aggregate line found" | tee -a "$LOG"
fi
