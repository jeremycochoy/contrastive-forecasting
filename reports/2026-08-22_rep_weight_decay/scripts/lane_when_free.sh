#!/bin/bash
# #409 — start one lane when the card has enough free VRAM.
#
# WHY THIS SCRIPT EXISTS. elisa's two cards already carry other agents' runs.
# This card must share them and must not stop another run. One leg of this
# cell holds about 5.4 GB: the latent-drift probe draws a 4.32 GB block at
# steps 0, 20,000 and 40,000, and the allocator keeps it. A lane that starts
# on a card with less free memory dies inside the probe in its first seconds,
# and `phase1.sh` then re-fires it twice more against the same wall.
#
# So this waits, reads the card, and starts the lane when the card can hold
# it. It never touches another process.
#
# The wait is a ceiling, not a budget. On a timeout the lane does NOT start,
# and the operator moves those arms to the other card.
#
# Usage:
#   ARMS="ctrl_s24 dec0_s24" GPU=0 nohup setsid bash scripts/lane_when_free.sh &
#
#   GPU         the card index. Default 0.
#   NEED_MIB    free VRAM this lane needs. Default 6500.
#   POLL        seconds between reads. Default 300.
#   MAX_WAIT    seconds before this gives up. Default 86400 (one day).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

GPU="${GPU:-0}"
NEED_MIB="${NEED_MIB:-6500}"
POLL="${POLL:-300}"
MAX_WAIT="${MAX_WAIT:-86400}"
ARMS="${ARMS:?usage: ARMS=\"<arm> ...\" GPU=0 bash lane_when_free.sh}"
mkdir -p "$CF409_RESULTS"

LOG="$CF409_RESULTS/lane_when_free_gpu${GPU}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409 wait gpu $GPU] $*" | tee -a "$LOG"; }

for arm in $ARMS; do cf409_require_arm "$arm" || exit $?; done
cf409_require_gpus "$GPU" || exit 2

log "arms='$ARMS' need=${NEED_MIB} MiB poll=${POLL}s max_wait=${MAX_WAIT}s"
waited=0
while :; do
  free="$(nvidia-smi --id="$GPU" --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  case "$free" in ''|*[!0-9]*) log "no reading from the driver — retrying"; free=0 ;; esac
  if [ "$free" -ge "$NEED_MIB" ]; then
    log "card has ${free} MiB free after ${waited}s — starting the lane"
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    log "GIVING UP after ${waited}s — the card holds ${free} MiB free," \
        "and this lane needs ${NEED_MIB}. Its arms did not start:" \
        "$ARMS"
    exit 1
  fi
  [ $(( waited % 1800 )) -eq 0 ] && log "${free} MiB free, need ${NEED_MIB}"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

ARMS="$ARMS" GPUS="$GPU" exec bash "$HERE/launch.sh"
