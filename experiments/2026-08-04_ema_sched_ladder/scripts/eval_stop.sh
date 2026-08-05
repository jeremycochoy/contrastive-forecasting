#!/bin/bash
# #393 — one head, one encoder, one stop.
#
# Usage: eval_stop.sh <cell_slug> <stop_steps> <student|teacher> \
#                     <head_steps> <score_out>
#        WT=<checkout> [RUNS=<checkpoint dir>] [BB_GPU=0] bash eval_stop.sh ...
#
# Trains a fresh 2L transformer quantile head on the cell's backbone
# checkpoint at <stop_steps>, then runs GIFT-Eval B4 over the 97 configs.
# Writes the aggregate GM-Relative MASE, and nothing else, to <score_out>.
#
# <student|teacher> selects the encoder for BOTH calls. The teacher is the
# EMA copy of the patch embedding and the encoder stack; --encoder-source
# loads it in the student's place, leaving the forecaster rollout on the
# student (the teacher has no forecaster). Head training records its
# encoder next to the checkpoint and the eval refuses the other one, so a
# teacher head cannot produce a student number.
#
# Protocol constants are the 2026-08-04 ones: head seed 20260722, forecast
# horizon 16, B4, 97 configs, and --grad-clip 1.0 on the head. The project
# bans grad clipping; the previous study kept it for comparability and so
# does this one, which the report has to say.
set -uo pipefail

CELL="${1:?usage: eval_stop.sh <cell> <stop> <student|teacher> <head_steps> <score_out>}"
STOP="${2:?stop steps}"
ENC="${3:?student|teacher}"
HEAD_STEPS="${4:?head steps}"
SCORE_OUT="${5:?score output path}"

case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac

WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
# A head is 15k-30k training steps and the GIFT-Eval outputs are its only
# record, so both live on the same durable root as the backbones rather
# than inside the checkout — see leg_paths.sh.
. "$(dirname "${BASH_SOURCE[0]}")/leg_paths.sh"
ROOT="$(runs_root)" || exit 2
RUNS="$ROOT/$CELL"
HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
GEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"

export PYTHONPATH="$WT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "ABORT: empty HF_TOKEN" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
STOP_K=$(( STOP / 1000 ))
NAME="cf393_${CELL}"
BB="$(ckpt_at_step "$RUNS" "$NAME" "$STOP_K")"
[ -n "$BB" ] || {
  echo "ABORT: no ${NAME}_${STOP_K}k.pth under $RUNS/leg_${STOP_K}k" >&2
  exit 3; }

OUT="$RUNS/eval/bb${STOP_K}k_${ENC}"
mkdir -p "$OUT" "$(dirname "$SCORE_OUT")"
HEAD_NAME="qhead_${NAME}_bb${STOP_K}k_${ENC}"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/eval.log"

# Backbone arch fields the state_dict cannot disambiguate. GIFT-Eval
# reconstructs the freq / seasonality embedding dims from the checkpoint,
# so it takes the shorter list.
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3 --encoder-type gru
           --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)
ARCH_EVAL=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3 --encoder-type gru
           --rev-norm-kind ewma --rev-norm-span 128)

echo "[$(date +%H:%M:%S)] $CELL bb${STOP_K}k enc=$ENC head=${HEAD_STEPS}s" | tee -a "$LOG"

if [ ! -f "$HEAD_CKPT" ]; then
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$HEAD_TRAIN" \
    --backbone-path "$BB" \
    --encoder-source "$ENC" \
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
  echo "[$(date +%H:%M:%S)] head-train rc=$rc" | tee -a "$LOG"
  [ $rc -eq 0 ] || exit $rc
else
  echo "[$(date +%H:%M:%S)] head-train SKIP (final exists)" | tee -a "$LOG"
fi

CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$GEVAL" \
  --backbone-path "$BB" \
  --head-path "$HEAD_CKPT" \
  --encoder-source "$ENC" \
  --output-dir "$OUT/gift" \
  --strategy B4 --forecast-len 16 --resume \
  "${ARCH_EVAL[@]}" \
  --head-nhead 8 --head-causal true \
  >>"$LOG" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] gift-eval rc=$rc" | tee -a "$LOG"
[ $rc -eq 0 ] || exit $rc

# The score, pinned to one file and one metric — see score_from_summary in
# leg_paths.sh for why neither is left to a glob.
SUMMARY="$OUT/gift/summary.txt"
score_from_summary "$SUMMARY" > "$SCORE_OUT.tmp" \
  || { rm -f "$SCORE_OUT.tmp"; exit 4; }
mv "$SCORE_OUT.tmp" "$SCORE_OUT"
echo "[$(date +%H:%M:%S)] DONE — $(grep -h 'Aggregate GM-Relative MASE' \
  "$SUMMARY") -> $(cat "$SCORE_OUT")" | tee -a "$LOG"
