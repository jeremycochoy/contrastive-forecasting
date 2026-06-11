#!/bin/bash
# On-instance chain: wait for backbone end -> promote checkpoints -> downstream
# best+last 2L & 6L in parallel (downstream_generic.sh, bottleneck 128).
set -u
TAG=allt08_xftrip_bn_enc6_qk_aon_b1024
RUNS=/root/out/runs
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain] $*"; }
say "waiting for backbone final.pth"
while true; do
  [ -f "$RUNS/bb_${TAG}_final.pth" ] && ! pgrep -f "train.py --resume" >/dev/null && break
  sleep 60
done
say "backbone finished; promoting checkpoints"
if [ -f "$RUNS/bb_${TAG}_best_loss.pth" ]; then cp -f "$RUNS/bb_${TAG}_best_loss.pth" "$RUNS/bb_${TAG}_FINAL.pth"
else cp -f "$RUNS/bb_${TAG}_final.pth" "$RUNS/bb_${TAG}_FINAL.pth"; fi
export WT=/root/cf-328 OUT=/root/out GIFT_EVAL=/root/gift-eval-data
chmod +x "$WT/experiments/2026-06-03_crossfade_triplet/scripts/downstream_generic.sh"
say "downstream start (2L+6L, parallel)"
bash "$WT/experiments/2026-06-03_crossfade_triplet/scripts/downstream_generic.sh" "$TAG" 2 0 128 > /root/out/results/dl2.log 2>&1 &
P2=$!
bash "$WT/experiments/2026-06-03_crossfade_triplet/scripts/downstream_generic.sh" "$TAG" 6 0 128 > /root/out/results/dl6.log 2>&1 &
P6=$!
wait $P2; say "downstream 2L rc=$?"
wait $P6; say "downstream 6L rc=$?"
say "CHAIN COMPLETE"
