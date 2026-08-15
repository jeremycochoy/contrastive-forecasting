#!/bin/bash
# #373 round 2 — wait for a card, then run `gap_jobs_r2.tsv` through the
# gap worker.
#
# Both of elisa's 4090s carry another session's training. The B5 backbone
# needs 5375 MiB (`results/gpu_mem_B5.csv`), and a card that is 300 MiB
# short kills the run at step 0 rather than making it wait. So the GPU is
# chosen when there is room for it, not when the script starts.
#
# Usage: bash gap_r2_launch.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$WT/reports/2026-08-08_rollout_depth/results"
mkdir -p "$RES"
LOG="$RES/gap_r2.log"

NEED="${NEED_MIB:-6200}"       # 5375 for the run, the rest is headroom.
TIMEOUT="${GPU_WAIT_TIMEOUT:-86400}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# The card with the most free memory, and how much it has.
freest(){
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | awk -F, '{ if ($2 + 0 > best + 0) { best = $2; g = $1 } } END { print g "," best }'
}

waited=0
while :; do
  read -r gpu free <<<"$(freest | tr ',' ' ')"
  [ -n "${free:-}" ] || { log "ABORT: nvidia-smi gave nothing"; exit 2; }
  [ "$free" -ge "$NEED" ] && break
  if [ "$waited" -ge "$TIMEOUT" ]; then
    log "TIMEOUT after ${waited}s: best card is GPU $gpu with ${free} MiB, need $NEED"
    exit 1
  fi
  [ $(( waited % 900 )) -eq 0 ] && log "waiting: GPU $gpu has ${free} MiB free, need $NEED"
  sleep 60; waited=$(( waited + 60 ))
done
log "GPU $gpu has ${free} MiB free after ${waited}s — starting the worker"

JOBS="$HERE/gap_jobs_r2.tsv" BB_GPU="$gpu" WT="$WT" \
  bash "$HERE/gap_worker.sh" r2 1 >>"$LOG" 2>&1
log "worker exited rc=$?"
