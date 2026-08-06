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
#
# HEAD_SEED overrides the head seed and nothing else. It exists for the
# replicate seeds: the extend rule reads a bb40k-to-bb100k difference, and
# for six of the ten cells that difference is smaller than the parent
# study's head-seed range, so a single seed cannot say whether the branch
# is a move or noise. A non-default seed files its head, its GIFT-Eval and
# its marker under `bb<N>k_<enc>_s<seed>` instead of `bb<N>k_<enc>`, so the
# seed-20260722 artefacts the ladder was decided from are never touched and
# the three seeds sit side by side. Everything else is identical: same
# backbone checkpoint, same encoder pairing, same head budget, same 97
# configs, same denominator.
#
# WHERE THE GIFT-EVAL RUNS is read from `results/EVAL_PLACE` at every stop,
# not fixed when the driver starts — the same trick as HOLD_ABOVE, and for
# the same reason: the placement was decided while ten cells were already
# climbing, and a driver that has been running for four hours has to pick
# it up. Three values:
#
#   inline     here, on this box's GPU. The original behaviour and the
#              default when the file is absent.
#   local_cpu  here, on this box's CPUs, sharded — elisa.
#   broker     on elisa. Write a request and wait for the score — vast.
#
# Head training is on the GPU in all three. Only the eval moves, because
# only the eval is CPU-bound: 100% of one core, a GPU 30% busy, 696 MiB of
# VRAM. On a vast box that GPU is rented and Exclusive_Process, so the eval
# holds the box's only CUDA context for hours while barely using it.
set -uo pipefail

CELL="${1:?usage: eval_stop.sh <cell> <stop> <student|teacher> <head_steps> <score_out>}"
STOP="${2:?stop steps}"
ENC="${3:?student|teacher}"
HEAD_STEPS="${4:?head steps}"
SCORE_OUT="${5:?score output path}"

case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac

# The protocol seed. A replicate passes HEAD_SEED and gets its own subtree.
HEAD_SEED_DEFAULT=20260722
HEAD_SEED="${HEAD_SEED:-$HEAD_SEED_DEFAULT}"
case "$HEAD_SEED" in
  ''|*[!0-9]*) echo "ABORT: HEAD_SEED='$HEAD_SEED' is not an integer" >&2; exit 2;;
esac
SEED_SUFFIX=""
[ "$HEAD_SEED" != "$HEAD_SEED_DEFAULT" ] && SEED_SUFFIX="_s${HEAD_SEED}"

# HEAD_TAG appends a further suffix to the eval subtree and the head name,
# for a rerun that changes something other than the seed. It exists for the
# GPU control: three cells' seed-20260722 bb40k heads were trained on a
# rented RTX 5090 and their bb40k replicates run on elisa's RTX 4090, so
# the spread over those three seeds would fold a hardware difference into a
# seed spread. Retraining seed 20260722 on the 4090 measures that
# difference directly, and it needs its own directory: the default-seed
# subtree already holds the 5090 head's GIFT-Eval, and eval_gift_eval's
# --resume would find all 97 configs done and re-emit the 5090 score.
# Empty by default, so every existing path is unchanged.
HEAD_TAG="${HEAD_TAG:-}"
if [ -n "$HEAD_TAG" ]; then
  case "$HEAD_TAG" in
    *[!a-zA-Z0-9_]*) echo "ABORT: HEAD_TAG='$HEAD_TAG' must be [a-zA-Z0-9_]" >&2; exit 2;;
  esac
  SEED_SUFFIX="${SEED_SUFFIX}_${HEAD_TAG}"
fi

# Exported, not just set: eval_local.sh runs as a child and falls back to a
# hard-coded elisa path when WT is absent from its environment, which on a
# rented box is a directory that does not exist. It only ever ran on elisa
# before, under eval_broker.sh, which passes WT explicitly.
export WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
# A head is 15k-30k training steps and the GIFT-Eval outputs are its only
# record, so both live on the same durable root as the backbones rather
# than inside the checkout — see leg_paths.sh.
. "$(dirname "${BASH_SOURCE[0]}")/leg_paths.sh"
. "$(dirname "${BASH_SOURCE[0]}")/gpu_gate.sh"
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

OUT="$RUNS/eval/bb${STOP_K}k_${ENC}${SEED_SUFFIX}"
mkdir -p "$OUT" "$(dirname "$SCORE_OUT")"

# A stop already scored costs nothing to re-enter. The ladder replays a
# cell from step 0 on every restart, and a `broker` eval is retried from
# the next tick when a push fails, so this path is walked more than once
# per stop by design.
if [ -s "$SCORE_OUT" ]; then
  echo "[$(date +%H:%M:%S)] $CELL bb${STOP_K}k $ENC SKIP — score $(cat "$SCORE_OUT")"
  exit 0
fi
HEAD_NAME="qhead_${NAME}_bb${STOP_K}k_${ENC}${SEED_SUFFIX}"
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

# One claim covers both CUDA processes below: the gate holds the device on
# fd 9 for the life of this shell, so the head train and the GIFT-Eval that
# reads its checkpoint are not separable by another cell.
gpu_gate "$BB_GPU" || { echo "ABORT: GPU $BB_GPU never came free" | tee -a "$LOG"; exit 1; }

echo "[$(date +%H:%M:%S)] $CELL bb${STOP_K}k enc=$ENC head=${HEAD_STEPS}s seed=$HEAD_SEED" | tee -a "$LOG"

if [ ! -f "$HEAD_CKPT" ]; then
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$HEAD_TRAIN" \
    --backbone-path "$BB" \
    --encoder-source "$ENC" \
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
  echo "[$(date +%H:%M:%S)] head-train rc=$rc" | tee -a "$LOG"
  [ $rc -eq 0 ] || exit $rc
else
  echo "[$(date +%H:%M:%S)] head-train SKIP (final exists)" | tee -a "$LOG"
fi

# The head is trained and the GPU's part of this stop is over. Drop the
# device claim before the eval: on `inline` the eval reclaims it, but on
# the other two placements the eval is elsewhere and holding the lock
# through a multi-hour wait would idle a rented GPU with a cell queued
# behind it.
exec 9>&- 2>/dev/null || true

PLACE_FILE="$WT/experiments/2026-08-04_ema_sched_ladder/results/EVAL_PLACE"
PLACE="inline"
[ -f "$PLACE_FILE" ] && PLACE="$(tr -dc 'a-z_' <"$PLACE_FILE")"
echo "[$(date +%H:%M:%S)] eval place=$PLACE" | tee -a "$LOG"

SUMMARY="$OUT/gift/summary.txt"

case "$PLACE" in

local_cpu)
  # elisa. Sharded across cores, one thread each, no GPU. eval_local.sh
  # writes SCORE_OUT itself, through the same score_from_summary as below.
  bash "$(dirname "${BASH_SOURCE[0]}")/eval_local.sh" \
    "$CELL" "$STOP_K" "$ENC" "$BB" "$HEAD_CKPT" "$OUT" "$SCORE_OUT" \
    >>"$LOG" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] eval_local rc=$rc" | tee -a "$LOG"
  [ $rc -eq 0 ] || exit $rc
  ;;

broker)
  # A vast box. Leave a request on disk and wait; scripts/eval_broker.sh on
  # elisa pulls the two checkpoints, evaluates, and pushes the score back.
  # The request is written to a temporary name and moved, because the
  # broker may read the directory at any moment.
  REQ="$OUT/EVAL_REQUEST"
  rm -f "$OUT/EVAL_TAKEN" "$OUT/EVAL_DONE"
  {
    echo "cell=$CELL"
    echo "stop_k=$STOP_K"
    echo "enc=$ENC"
    echo "backbone=$BB"
    echo "head=$HEAD_CKPT"
    echo "out_dir=$OUT"
    echo "score_out=$SCORE_OUT"
    echo "created=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "$REQ.tmp" && mv -f "$REQ.tmp" "$REQ"
  mkdir -p "$OUT/gift"
  echo "[$(date +%H:%M:%S)] request written, waiting for the broker" | tee -a "$LOG"

  # 12 h. One eval is ~3 core-hours split over the shards, so this covers a
  # deep queue on elisa. Failing instead of waiting would throw away a head
  # that already cost 15,000 or 30,000 GPU steps.
  BROKER_WAIT="${BROKER_WAIT:-43200}"
  waited=0
  while [ ! -s "$SCORE_OUT" ]; do
    if [ "$waited" -ge "$BROKER_WAIT" ]; then
      echo "[$(date +%H:%M:%S)] ABORT: no score after ${waited}s" | tee -a "$LOG"
      exit 8
    fi
    [ $(( waited % 1800 )) -eq 0 ] && [ "$waited" -gt 0 ] && \
      echo "[$(date +%H:%M:%S)] still waiting for the broker (${waited}s)" | tee -a "$LOG"
    sleep 60
    waited=$(( waited + 60 ))
  done
  echo "[$(date +%H:%M:%S)] broker returned after ${waited}s" | tee -a "$LOG"
  ;;

*)
  # inline — this box's GPU, unsharded. Reclaim the device first.
  gpu_gate "$BB_GPU" || { echo "ABORT: GPU $BB_GPU never came free" | tee -a "$LOG"; exit 1; }
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
  ;;
esac

# The score, pinned to one file and one metric — see score_from_summary in
# leg_paths.sh for why neither is left to a glob. `local_cpu` and `broker`
# have already written it; `inline` has not.
if [ ! -s "$SCORE_OUT" ]; then
  score_from_summary "$SUMMARY" > "$SCORE_OUT.tmp" \
    || { rm -f "$SCORE_OUT.tmp"; exit 4; }
  mv "$SCORE_OUT.tmp" "$SCORE_OUT"
fi
echo "[$(date +%H:%M:%S)] DONE — $(grep -h 'Aggregate GM-Relative MASE' \
  "$SUMMARY" 2>/dev/null) -> $(cat "$SCORE_OUT")" | tee -a "$LOG"
