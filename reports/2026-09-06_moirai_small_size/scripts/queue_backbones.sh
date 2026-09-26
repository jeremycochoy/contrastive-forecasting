#!/bin/bash
# #412 — the backbones that wait for a card, in one queue on one card.
#
# `phase1.sh` runs its arms ONE AT A TIME: it waits for each leg, then trains
# a head on the same card. Two arms fit on one card at this width, and the
# heads belong on the other card, so this queue does neither.
#
# WHAT IT DOES. It takes an ordered list of arms. For each arm it waits for
# the free memory that arm needs, starts `run_arm.sh` in the background, waits
# for the trainer to hold the card, then moves to the next arm. So the arms
# run TOGETHER, and the next arm reads a free number that already counts the
# arm before it.
#
# WHY THE ORDER IS BY MEMORY. The first arm of the list takes the first window
# that fits it. An arm that needs 11,300 MiB behind an arm that needs 7,700
# never starts, because the small arm takes the window and leaves 4,000. So
# the list runs from the largest need to the smallest.
#
# WHAT IT DOES NOT DO. It trains no head. `missing_heads.sh` sweeps the other
# card every 15 minutes and starts the head of every backbone that has none.
#
# Usage:  BB_GPU=0 CF412_QUEUE="k32_r200_08 k8_r100_09" \
#           bash scripts/queue_backbones.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

GPU="${BB_GPU:-0}"
QUEUE="${CF412_QUEUE:?usage: CF412_QUEUE=\"<arm> <arm>\" queue_backbones.sh}"
STOP="${CF412_QUEUE_STOP:-40000}"
SETTLE="${CF412_QUEUE_SETTLE:-300}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 queue] $*" \
  | tee -a "$CF412_RESULTS/queue.log"; }

for arm in $QUEUE; do cf412_require_arm "$arm" || exit $?; done
cf412_require_stop "$STOP" || exit $?
log "gpu $GPU  stop $STOP  queue: $QUEUE"

for arm in $QUEUE; do
  if [ -f "$(cf412_collapse_file "$arm")" ]; then
    log "$arm SKIPPED — it lost the contrastive task in an earlier pass"
    continue
  fi
  need="$(cf412_leg_vram_mib "$arm")"
  log "$arm waits for ${need} MiB free on gpu $GPU"
  cf412_wait_for_vram "$GPU" "$need" "backbone $arm" \
    2>&1 | tee -a "$CF412_RESULTS/queue.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    log "$arm SKIPPED — gpu $GPU never had ${need} MiB free"; continue
  fi
  log "$arm -> $STOP steps on gpu $GPU"
  BB_GPU="$GPU" nohup bash "$HERE/run_arm.sh" "$arm" "$STOP" \
    >>"$CF412_RESULTS/queue_${arm}.log" 2>&1 &
  child=$!
  log "$arm run_arm pid $child"
  # The next arm must read a free number that counts this one. The trainer
  # allocates over its first minutes, so the queue waits for the process to
  # hold the card and then for the allocation to settle.
  waited=0
  while [ "$waited" -lt "$SETTLE" ]; do
    sleep 30; waited=$(( waited + 30 ))
    kill -0 "$child" 2>/dev/null || { log "$arm run_arm exited early"; break; }
  done
  free="$(nvidia-smi --id="$GPU" --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  log "$arm settled — gpu $GPU has ${free:-?} MiB free"
done
log "queue done"
