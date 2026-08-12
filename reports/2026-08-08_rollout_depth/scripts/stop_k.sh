#!/bin/bash
# #373 — one head and one GIFT-Eval, for one (cell, k, stop, encoder).
#
# The protocol is #393's, unchanged: quantile head, 2-layer transformer,
# forecast-len 16, batch 256, lr 1e-3, --grad-clip 1.0 on the head, head
# seed 20260722, 15,000 steps at bb40k and 30,000 from bb100k, then the 97
# GIFT-Eval configs under the official B4 strategy. Head training on the
# GPU, GIFT-Eval on elisa's cores — the split PR #394 decided, and the
# reason 14 cells are affordable at all here.
#
# What is NOT #393's: which checkpoint it reads. This study's two arms of a
# cell differ only in k, and their run names differ only in the `_cf373k<K>`
# suffix, so the resolution goes through cell_paths.sh.
#
# Usage: stop_k.sh <cell id> <k> <stop steps> <student|teacher>
set -uo pipefail

CELL_ID="${1:?usage: stop_k.sh <cell id> <k> <stop steps> <student|teacher>}"
K="${2:?k}"
STOP="${3:?stop steps}"
ENC="${4:?student|teacher}"
case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac

HEAD_SEED="${HEAD_SEED:-20260722}"
STOP_K=$(( STOP / 1000 ))
# #393's budget: 15k at bb40k, 30k from bb100k.
if [ "$STOP" -le 40000 ]; then HEAD_STEPS="${HEAD_STEPS:-15000}"
else                           HEAD_STEPS="${HEAD_STEPS:-30000}"; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$WT/reports/2026-08-08_rollout_depth/results"
mkdir -p "$RES"
. "$HERE/cell_paths.sh"
. "$HERE/gpu_gate.sh"

BB="$(cf373_bb_ckpt "$CELL_ID" "$K" "$STOP")"
[ -n "$BB" ] && [ -f "$BB" ] || {
  echo "ABORT: no bb${STOP_K}k checkpoint for $CELL_ID k=$K under $(cf373_runs_dir "$CELL_ID" "$K" "$STOP")" >&2
  exit 3; }

TAG="${CELL_ID}_k${K}_bb${STOP_K}k_${ENC}"
OUT="$CF373_ROOT/eval/$TAG"
SCORE_OUT="$RES/score_${TAG}.txt"
HEAD_NAME="qhead_${TAG}_s${HEAD_SEED}"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/stop.log"
mkdir -p "$OUT"

HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
[ -f "$HEAD_TRAIN" ] || { echo "ABORT: no head trainer at $HEAD_TRAIN" >&2; exit 2; }
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "ABORT: empty HF_TOKEN" >&2; exit 2; }

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$TAG] $*" | tee -a "$LOG" "$RES/stops.log"; }

if [ -s "$SCORE_OUT" ]; then
  log "SKIP — already scored $(cat "$SCORE_OUT")"; exit 0
fi

# Arch fields the state_dict cannot disambiguate. GIFT-Eval rebuilds the
# freq / seasonality dims from the checkpoint, so it takes the shorter list.
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3 --encoder-type gru
           --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)

# Wait for free VRAM before starting a head, and hold the wait on a lock so
# two of this study's own heads cannot both pass the check and then both
# allocate.
#
# `gpu_gate` does not cover this: it returns immediately on a `Default`-mode
# card, which is what elisa and every box here run. The head trainer asks for
# a 4.32 GiB block inside the GRU encoder, and elisa's GPU 1 is shared with
# another session whose job grew from 9 to 14 GiB mid-study — three stops
# died on `torch.OutOfMemoryError` inside `_last_hidden` before this existed.
head_vram_gate(){ # <gpu index>
  local gpu="$1" need="${HEAD_VRAM_MIB:-7000}" waited=0 free
  local lock="${GPU_GATE_LOCKDIR:-/tmp}/cf373_head_gpu${gpu}.lock"
  : >>"$lock" 2>/dev/null || true
  exec 7>>"$lock" || return 0
  flock -w 86400 7 || { log "timed out waiting for the head lock"; return 1; }
  while :; do
    free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$free" ] || return 0            # no nvidia-smi: proceed ungated
    [ "$free" -ge "$need" ] && break
    if [ "$waited" -ge "${HEAD_VRAM_TIMEOUT:-14400}" ]; then
      log "TIMEOUT after ${waited}s: GPU $gpu has ${free} MiB free, need ${need}"
      return 1
    fi
    [ $(( waited % 600 )) -eq 0 ] && \
      log "waiting for VRAM on GPU $gpu: ${free} MiB free, need ${need}"
    sleep 30; waited=$(( waited + 30 ))
  done
  [ "$waited" -gt 0 ] && log "GPU $gpu has ${free} MiB free after ${waited}s"
  return 0
}

if [ ! -f "$HEAD_CKPT" ]; then
  BB_GPU="${BB_GPU:-1}"
  gpu_gate "$BB_GPU" || { log "ABORT: GPU $BB_GPU never came free"; exit 1; }
  head_vram_gate "$BB_GPU" || { log "ABORT: not enough VRAM on GPU $BB_GPU"; exit 1; }
  log "head-train start enc=$ENC steps=$HEAD_STEPS seed=$HEAD_SEED gpu=$BB_GPU bb=$(basename "$BB")"
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
  log "head-train rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  # Drop the device before the eval: it runs on the CPU, and holding a lock
  # through a multi-hour CPU job idles a card with a cell queued. BOTH
  # descriptors, not just the gpu_gate one — fd 7 is the head VRAM lock, and
  # leaving it open kept the next head waiting for a whole GIFT-Eval.
  exec 9>&- 2>/dev/null || true
  exec 7>&- 2>/dev/null || true
else
  log "head-train SKIP (final exists)"
fi

log "eval start (97 configs, B4, forecast-len 16, elisa CPUs)"
bash "$HERE/eval_local.sh" "$TAG" "$STOP_K" "$ENC" "$BB" "$HEAD_CKPT" \
  "$OUT" "$SCORE_OUT" >>"$LOG" 2>&1
rc=$?
log "eval rc=$rc"
[ $rc -eq 0 ] || exit $rc
log "DONE — GM-Relative MASE $(cat "$SCORE_OUT")"
