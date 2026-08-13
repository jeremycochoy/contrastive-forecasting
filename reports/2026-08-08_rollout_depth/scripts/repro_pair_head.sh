#!/bin/bash
# #373 — retrain ONE quantile head from an EXPLICIT backbone file.
#
# Why this exists, beside r2_head_box.sh. The card asked whether A1 and B3
# share a student score because the eval path drops the EMA regime. That
# question cannot be answered by the same code path that produced the
# number: r2_head_box.sh resolves its checkpoint through cell_paths.sh, so a
# resolution fault would repeat itself and reproduce the same number twice.
#
# This script takes the checkpoint as a literal path, prints its md5 before
# training, and writes that md5 beside the head. Two cells that resolve to
# different md5s and still produce the same score prove the score follows
# the weights, not the path.
#
# Protocol is r2_head_box.sh's, unchanged: quantile head, 2-layer
# transformer, forecast-len 16, batch 256, lr 1e-3, --grad-clip 1.0,
# seed 20260722, 15,000 steps off a 40k backbone and 30,000 off 100k.
#
# Usage: repro_pair_head.sh <tag> <backbone.pth> <student|teacher> <steps> <gpu>
set -uo pipefail

TAG="${1:?usage: repro_pair_head.sh <tag> <bb.pth> <student|teacher> <steps> <gpu>}"
BB="${2:?backbone path}"
ENC="${3:?student|teacher}"
HEAD_STEPS="${4:?steps}"
HEAD_GPU="${5:-1}"

case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac
[ -f "$BB" ] || { echo "ABORT: no backbone at $BB" >&2; exit 3; }

WT="${WT:-/root/cf}"
export CF373_ROOT="${CF373_ROOT:-/root/cf373_runs}"
HEAD_SEED="${HEAD_SEED:-20260722}"

OUT="$CF373_ROOT/eval/$TAG"
HEAD_NAME="qhead_${TAG}_s${HEAD_SEED}"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/head.log"
mkdir -p "$OUT"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [repro $TAG] $*" | tee -a "$LOG"; }

BB_MD5="$(md5sum "$BB" | cut -d' ' -f1)"
printf '%s\n' "$(basename "$BB")" > "$OUT/backbone.txt"
printf '%s\n' "$BB_MD5"           > "$OUT/backbone_md5.txt"
printf '%s\n' "$BB"               > "$OUT/backbone_path.txt"

if [ -f "$HEAD_CKPT" ]; then
  log "SKIP — $(basename "$HEAD_CKPT") exists (bb md5 $BB_MD5)"
  exit 0
fi

HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
[ -f "$HEAD_TRAIN" ] || { log "ABORT: no head trainer at $HEAD_TRAIN"; exit 2; }
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3 --encoder-type gru
           --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)

# Same VRAM gate as r2_head_box.sh, same per-card lock, so a repro head and
# a queue head on one card wait for each other instead of racing to OOM.
need="${HEAD_VRAM_MIB:-8000}"; waited=0
lock="/tmp/cf373_r2_head.gpu$HEAD_GPU.lock"
: >>"$lock" 2>/dev/null || true
exec 7>>"$lock" && flock -w 86400 7
while :; do
  free=$(nvidia-smi --id="$HEAD_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$free" ] || break
  [ "$free" -ge "$need" ] && break
  if [ "$waited" -ge "${HEAD_VRAM_TIMEOUT:-21600}" ]; then
    log "ABORT: ${free} MiB free after ${waited}s, need $need"; exit 1; fi
  [ $(( waited % 600 )) -eq 0 ] && log "waiting for VRAM: ${free} MiB free, need $need"
  sleep 30; waited=$(( waited + 30 ))
done

log "start enc=$ENC steps=$HEAD_STEPS seed=$HEAD_SEED gpu=$HEAD_GPU bb=$(basename "$BB") md5=$BB_MD5"
CUDA_VISIBLE_DEVICES="$HEAD_GPU" python3 -u "$HEAD_TRAIN" \
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
  "${ARCH_HEAD[@]}" >>"$LOG" 2>&1 7>&- &
train_pid=$!
sleep 180; exec 7>&- 2>/dev/null || true
wait "$train_pid"; rc=$?
log "rc=$rc"
[ $rc -eq 0 ] || exit $rc
[ -f "$HEAD_CKPT" ] || { log "ABORT: rc=0 but no $HEAD_CKPT"; exit 4; }
log "DONE $(du -h "$HEAD_CKPT" | cut -f1)  bb md5 $BB_MD5"
