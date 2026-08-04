#!/bin/bash
# #390 — 2L quantile head + GIFT-Eval B4 on one arm's backbone snapshot.
#
# The measurement half of a wave. #379's `eval_2L_gm_mase.sh` verbatim except
# for three things it hardcodes and #390 cannot share: the checkout, the
# experiment dir, and the arm → run-name resolution (that one now comes from
# `arm_names.sh` instead of an awk over run_arm.sh's case block).
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train ARM=arm5 BB_GPU=0 \
#     BB_STEP_K=40 HEAD_STEPS=15000 bash eval_arm.sh
#
# The wave decides (BB_STEP_K, HEAD_STEPS):
#   wave 1 |  40k backbone, 15 000-step head
#   wave 2 | 100k backbone, 30 000-step head  (fresh head, new output dir)
#   wave 3 | 200k backbone, 30 000-step head  (fresh head, new output dir)
#
# A head is "fresh" because the output dir is keyed on the cell name, which
# carries both numbers — nothing from an earlier wave is on that path to
# resume from.
#
# GIFT-Eval must cover all 97 B4 configs. `--resume` fills in from the
# partial all_results.csv, so a killed eval continues rather than restarting;
# this script retries until the row count is 97 and exits non-zero if it
# never gets there. It never reports a partial.
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

ARM="${ARM:?set ARM=<slug>}"
BB_GPU="${BB_GPU:?set BB_GPU=0 or 1}"
BB_STEP_K="${BB_STEP_K:-40}"
HEAD_STEPS="${HEAD_STEPS:-15000}"
N_CONFIGS_EXPECTED=97
MAX_EVAL_TRIES="${MAX_EVAL_TRIES:-3}"
# Two knobs the review added, both defaulted to what every wave already ran:
#   HEAD_SEED — the head's init/data seed. Review item 4 measures the spread
#               over seeds; the waves all used 20260722, #379's value.
#   CELL_TAG  — inserted into the cell name, so a replicate seed or the
#               student control gets its own output directory instead of
#               resuming the wave's head off disk.
HEAD_SEED="${HEAD_SEED:-20260722}"
CELL_TAG="${CELL_TAG:-}"

EXP="$WT/experiments/2026-08-01_lalign_teacher"
RUNS="$EXP/runs"
OUT_ROOT="$EXP/eval_gm_mase"
HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
GEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"

export PYTHONPATH="$WT"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"

[ -f "$HEAD_TRAIN" ] || { echo "ABORT: head trainer not at $HEAD_TRAIN" >&2; exit 2; }
[ -f "$GEVAL" ]      || { echo "ABORT: gift-eval not at $GEVAL" >&2; exit 2; }
[ -n "$HF_TOKEN" ]   || { echo "ABORT: empty HF_TOKEN" >&2; exit 2; }

NAME="$(bb_name "$ARM")" || exit 2

# The snapshot for this wave. A resumed run writes a fresh `_r<N>` safe run
# name, so match those too, newest first.
BB=$(ls -t "$RUNS/${NAME}"_${BB_STEP_K}k.pth "$RUNS/${NAME}"_r*_${BB_STEP_K}k.pth 2>/dev/null | head -1)
[ -f "$BB" ] || { echo "ABORT: no ${BB_STEP_K}k backbone for arm '$ARM' (name=$NAME)" >&2; exit 3; }

CELL="${ARM}${CELL_TAG}_bb${BB_STEP_K}k_hd${HEAD_STEPS}s"
OUT="$OUT_ROOT/$CELL"; mkdir -p "$OUT"
HEAD_NAME="qhead_2L_${NAME}_bb${BB_STEP_K}k"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/eval.log"
CSV="$OUT/gift/all_results.csv"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [$CELL] $*" | tee -a "$LOG"; }

# Data rows in all_results.csv (header excluded); 0 when the file is absent.
n_rows(){ [ -f "$CSV" ] && awk 'END{print (NR>0 ? NR-1 : 0)}' "$CSV" || echo 0; }

say "start on GPU $BB_GPU (backbone=$(basename "$BB"))"

# Backbone arch args accepted by train_forecasting_head.py (it reads the
# remaining backbone flags out of the checkpoint config).
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)
# GIFT-Eval reconstructs freq/seasonality embedding dims from the checkpoint
# and rejects the flags; everything else it takes.
ARCH_EVAL=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3
           --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

# --- 1) fresh 2L transformer quantile head ---------------------------------
if [ ! -f "$HEAD_CKPT" ]; then
  say "head-train ${HEAD_STEPS} steps"
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
  say "head-train rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  [ -f "$HEAD_CKPT" ] || { say "ABORT: head trainer exited 0 but wrote no $HEAD_CKPT"; exit 4; }
else
  say "head-train SKIP (final head already on disk)"
fi

# --- 2) GIFT-Eval B4, all 97 configs ---------------------------------------
# `--resume` reads the partial CSV, so a retry costs only the missing configs.
try=1
while [ "$try" -le "$MAX_EVAL_TRIES" ]; do
  have=$(n_rows)
  if [ "$have" -ge "$N_CONFIGS_EXPECTED" ]; then break; fi
  say "gift-eval try $try/$MAX_EVAL_TRIES (have $have/$N_CONFIGS_EXPECTED configs)"
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$GEVAL" \
    --backbone-path "$BB" \
    --head-path "$HEAD_CKPT" \
    --output-dir "$OUT/gift" \
    --strategy B4 --forecast-len 16 --resume \
    "${ARCH_EVAL[@]}" \
    --head-nhead 8 --head-causal true \
    >>"$LOG" 2>&1
  say "gift-eval try $try rc=$? rows=$(n_rows)"
  try=$((try + 1))
done

have=$(n_rows)
if [ "$have" -ne "$N_CONFIGS_EXPECTED" ]; then
  say "FAILED — $have/$N_CONFIGS_EXPECTED configs after $MAX_EVAL_TRIES tries. No partial reported."
  exit 5
fi

# Aggregate: the eval script tees one `Aggregate GM-Relative MASE (N configs)`
# line into its own summary.txt. Lift it to the cell root and one level up, so
# a wave's numbers are one `cat` away.
AGG=$(grep -h "Aggregate GM-Relative MASE" "$OUT/gift/summary.txt" 2>/dev/null | tail -1)
if [ -z "$AGG" ]; then
  say "FAILED — 97 rows on disk but no Aggregate line in gift/summary.txt"
  exit 6
fi
case "$AGG" in
  *"($N_CONFIGS_EXPECTED configs)"*) : ;;
  *) say "FAILED — aggregate is not over $N_CONFIGS_EXPECTED configs: $AGG"; exit 7 ;;
esac
echo "$AGG" > "$OUT/summary.txt"
echo "$AGG" > "$OUT_ROOT/${CELL}_summary.txt"
say "DONE — $AGG"
