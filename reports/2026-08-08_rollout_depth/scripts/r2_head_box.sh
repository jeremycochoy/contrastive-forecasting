#!/bin/bash
# #373 round 2 — train ONE quantile head, on the rented box that trained
# the backbone.
#
# Round 1 trained heads on elisa, because the study had $7.31 and elisa's
# cards were free. Round 2 has $100 and elisa's two cards are full: 23.1 GB
# and 22.7 GB of 24.5 GB were held by other sessions when this round was
# planned, and one head needs ~7 GB. So the head follows the backbone onto
# the card that just produced it. The GIFT-Eval still runs on elisa's cores,
# for the reason bootstrap_remote.sh gives: one machine produces every
# GM-Relative MASE in this study, so no rented card can put one on a
# different scale.
#
# The protocol is #393's, and it is stop_k.sh's head block byte for byte:
# quantile head, 2-layer transformer, forecast-len 16, batch 256, lr 1e-3,
# --grad-clip 1.0, head seed 20260722, 15,000 steps at bb40k and 30,000
# from bb100k.
#
# Usage (on the box):  r2_head_box.sh <cell> <k> <stop steps> <student|teacher>
set -uo pipefail

CELL="${1:?usage: r2_head_box.sh <cell> <k> <stop> <student|teacher>}"
K="${2:?k}"
STOP="${3:?stop steps}"
ENC="${4:?student|teacher}"
case "$ENC" in student|teacher) ;; *) echo "ABORT: bad encoder '$ENC'" >&2; exit 2;; esac

WT="${WT:-/root/cf}"
HERE="$WT/reports/2026-08-08_rollout_depth/scripts"
RES="$WT/reports/2026-08-08_rollout_depth/results"
export CF373_ROOT="${CF373_ROOT:-/root/cf373_runs}"
mkdir -p "$RES"
. "$HERE/cell_paths.sh"

HEAD_SEED="${HEAD_SEED:-20260722}"
STOP_K=$(( STOP / 1000 ))
if [ "$STOP" -le 40000 ]; then HEAD_STEPS="${HEAD_STEPS:-15000}"
else                           HEAD_STEPS="${HEAD_STEPS:-30000}"; fi

TAG="${CELL}_k${K}_bb${STOP_K}k_${ENC}"
OUT="$CF373_ROOT/eval/$TAG"
HEAD_NAME="qhead_${TAG}_s${HEAD_SEED}"
HEAD_CKPT="$OUT/${HEAD_NAME}_final.pth"
LOG="$OUT/head.log"
mkdir -p "$OUT"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [head $TAG] $*" | tee -a "$LOG" "$RES/heads.log"; }

if [ -f "$HEAD_CKPT" ]; then log "SKIP — $(basename "$HEAD_CKPT") exists"; exit 0; fi

BB="$(cf373_bb_ckpt "$CELL" "$K" "$STOP")"
[ -n "$BB" ] && [ -f "$BB" ] || {
  log "ABORT: no bb${STOP_K}k checkpoint under $(cf373_runs_dir "$CELL" "$K" "$STOP")"
  exit 3; }

HEAD_TRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
[ -f "$HEAD_TRAIN" ] || { log "ABORT: no head trainer at $HEAD_TRAIN"; exit 2; }
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 2; }

# Arch fields the state_dict cannot disambiguate. GIFT-Eval rebuilds the
# freq / seasonality dims from the checkpoint, so it takes the shorter list.
ARCH_HEAD=(--t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
           --num-layers 3 --encoder-type gru
           --rev-norm-kind ewma --rev-norm-span 128
           --freq-emb-dim 3 --seasonality-emb-dim 3)

# The box runs the backbone wave and up to two heads on one card at once.
# Three processes fit — 5.4 GB for the backbone and ~7 GB per head against
# 32 GB — but only if they do not all allocate in the same instant. Wait for
# the room, under a lock, exactly as stop_k.sh does on elisa.
need="${HEAD_VRAM_MIB:-8000}"; waited=0
lock=/tmp/cf373_r2_head.lock
: >>"$lock" 2>/dev/null || true
exec 7>>"$lock" && flock -w 86400 7
while :; do
  free=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$free" ] || break
  [ "$free" -ge "$need" ] && break
  if [ "$waited" -ge "${HEAD_VRAM_TIMEOUT:-21600}" ]; then
    log "ABORT: ${free} MiB free after ${waited}s, need $need"; exit 1; fi
  [ $(( waited % 600 )) -eq 0 ] && log "waiting for VRAM: ${free} MiB free, need $need"
  sleep 30; waited=$(( waited + 30 ))
done
[ "$waited" -gt 0 ] && log "got VRAM after ${waited}s"

log "start enc=$ENC steps=$HEAD_STEPS seed=$HEAD_SEED bb=$(basename "$BB")"
CUDA_VISIBLE_DEVICES=0 python3 -u "$HEAD_TRAIN" \
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
  "${ARCH_HEAD[@]}" >>"$LOG" 2>&1 &
train_pid=$!
# Release the lock once the process is past its allocation ramp, so the
# second head can start while this one trains. Holding it for the whole
# 15,000 steps would serialise the two heads for no reason.
sleep 180; exec 7>&- 2>/dev/null || true
wait "$train_pid"; rc=$?
log "rc=$rc"
[ $rc -eq 0 ] || exit $rc
[ -f "$HEAD_CKPT" ] || { log "ABORT: rc=0 but no $HEAD_CKPT"; exit 4; }
# The eval reads the backbone from elisa's copy, so the head has to be
# pairable with it. Record which checkpoint produced it.
printf '%s\n' "$(basename "$BB")" > "$OUT/backbone.txt"
log "DONE $(du -h "$HEAD_CKPT" | cut -f1)"
