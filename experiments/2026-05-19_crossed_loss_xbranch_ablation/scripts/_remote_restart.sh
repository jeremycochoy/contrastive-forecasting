#!/bin/bash
# Runs ON the box. Kill any prior backbone, clean partial artifacts,
# relaunch box_run backbone <arm> detached, authenticated.
#   _remote_restart.sh <arm>
set -u
A="${1:?arm}"
cd /workspace/app
pkill -9 -f 'box_run.sh' 2>/dev/null || true
pkill -9 -f 'torchrun'   2>/dev/null || true
pkill -9 -f 'train.py'   2>/dev/null || true
sleep 5
rm -rf /workspace/app/runs /workspace/app/results /workspace/app/box_*.log
export HF_TOKEN="$(cat /workspace/app/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "restart arm=$A token=$(wc -c < /workspace/app/experiments/hf_token.txt)B $(date -u +%FT%TZ)"
setsid nohup bash experiments/2026-05-19_crossed_loss_xbranch_ablation/scripts/box_run.sh \
  backbone "$A" > "/workspace/app/box_${A}_backbone.log" 2>&1 < /dev/null &
disown 2>/dev/null || true
sleep 3
echo "relaunched pid=$(pgrep -f "box_run.sh backbone $A" | head -1)"
