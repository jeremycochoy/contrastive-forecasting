#!/bin/bash
# #373 — one head and one GIFT-Eval on an EXPLICIT backbone path.
#
# `stop_k.sh` resolves its backbone from (cell, k, stop) through
# cell_paths.sh. The review gaps need heads on backbones that no (cell, k)
# names: a PUBLISHED parent checkpoint, and a second backbone seed. Rather
# than teach cell_paths.sh about them, this takes the path.
#
# Everything else is stop_k.sh's, unchanged and on purpose: same head
# trainer, same 15,000 steps, same head seed 20260722, same --grad-clip 1.0,
# same ARCH lists, same 97-config B4 GIFT-Eval on elisa's cores. The whole
# point of gap 1 is that only the BACKBONE differs from a run of this study.
#
# Usage: head_eval_bb.sh <tag> <backbone .pth> <student|teacher> [head steps]
set -uo pipefail

TAG="${1:?usage: head_eval_bb.sh <tag> <backbone> <student|teacher> [steps]}"
BB="${2:?backbone .pth}"
ENC="${3:?student|teacher}"
HEAD_STEPS="${4:-15000}"
case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac
[ -f "$BB" ] || { echo "ABORT: no backbone at $BB" >&2; exit 3; }

HEAD_SEED="${HEAD_SEED:-20260722}"

# The backbone stop, in thousands, for the eval's log lines only. #373 runs
# every head of this script on a bb40k backbone, so 40 keeps its logs
# unchanged. #401 puts heads on 40k, 100k and 200k stops, and a wrong label
# in `results/stops.log` misnames the number a report reads.
CF_STOP_K="${CF_STOP_K:-40}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
# Which study's directory takes the head log and the score file. #401 reuses
# this head and this eval on its own backbones and points it at its own
# directory. Unset, this is #373's own directory, unchanged.
#
# CF_RESULTS names that results directory OUTRIGHT, for a caller whose
# results do not sit at `<study>/results` — #401's trial runs write to
# `<study>/results/trial`, and a score file that landed one level up would be
# collected as a real one.
RES="${CF_RESULTS:-${CF_STUDY_DIR:-$WT/reports/2026-08-08_rollout_depth}/results}"
mkdir -p "$RES"
. "$HERE/cell_paths.sh"
. "$HERE/gpu_gate.sh"

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

# stop_k.sh's list, verbatim.
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3 --encoder-type gru
           --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)

head_vram_gate(){ # <gpu index>
  local gpu="$1" need="${HEAD_VRAM_MIB:-7000}" waited=0 free
  local lock="${GPU_GATE_LOCKDIR:-/tmp}/cf373_head_gpu${gpu}.lock"
  : >>"$lock" 2>/dev/null || true
  exec 7>>"$lock" || return 0
  flock -w 86400 7 || { log "timed out waiting for the head lock"; return 1; }
  while :; do
    free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$free" ] || return 0
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
  BB_GPU="${BB_GPU:-0}"
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
  # Release both device claims before the CPU eval. See stop_k.sh.
  exec 9>&- 2>/dev/null || true
  exec 7>&- 2>/dev/null || true
else
  log "head-train SKIP (final exists)"
fi

log "eval start (97 configs, B4, forecast-len 16, elisa CPUs)"
bash "$HERE/eval_local.sh" "$TAG" "$CF_STOP_K" "$ENC" "$BB" "$HEAD_CKPT" \
  "$OUT" "$SCORE_OUT" >>"$LOG" 2>&1
rc=$?
log "eval rc=$rc"
[ $rc -eq 0 ] || exit $rc
log "DONE — GM-Relative MASE $(cat "$SCORE_OUT")"
