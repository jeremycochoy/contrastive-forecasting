#!/bin/bash
# #393 — climb several cells one after another on a vast.ai box.
#
# Usage, on the remote:  bash queue_remote.sh <cell> [cell ...]
#
# Elisa runs two cells side by side on one GPU. A vast.ai box cannot: it
# comes up in `Exclusive_Process` compute mode and the container cannot
# change it (`nvidia-smi -c 0` → "Insufficient Permissions"), so a second
# CUDA context dies at `.to(device)` with "CUDA-capable device(s) is/are
# busy or unavailable". This queue is what that constraint costs — nothing
# else. A cell is already a strict sequence (train a leg, train a head,
# evaluate, repeat), so serialising cells wastes no GPU time; it only
# delays when the later cells' numbers arrive.
#
# Waits on `ladder.py` processes, not on its own children: a driver left
# over from an earlier launch owns the GPU just as much.
set -uo pipefail

EXP=/root/cf/experiments/2026-08-04_ema_sched_ladder
cd "$EXP" || exit 2
mkdir -p results

for cell in "$@"; do
  while pgrep -f "ladder.py --cells" >/dev/null; do sleep 60; done
  echo "[queue] $(date '+%m-%d %H:%M:%S') start $cell"
  WT=/root/cf RUNS=/root/cf393_runs BB_GPU=0 \
  GIFT_EVAL=/root/workspaces/gift-eval-data \
    python3 -u scripts/ladder.py --cells "$cell" \
    >> "results/ladder_${cell}.log" 2>&1
  echo "[queue] $(date '+%m-%d %H:%M:%S') $cell exited rc=$?"
done
echo "[queue] $(date '+%m-%d %H:%M:%S') all cells done"
