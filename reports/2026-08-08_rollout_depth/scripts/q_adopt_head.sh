#!/bin/bash
# #373 — hand a head that ran OUTSIDE the queue back to the dispatcher.
#
# The rented box is gone, so the queue's four slots are two dead `rem:`
# entries and elisa's two cards. elisa's cards carry other projects, and
# `q_run.sh` gates a head on 8500 MiB free. A4's teacher head needed a card
# when 8426 MiB were free — 74 MiB under the gate — and the dispatcher's
# next choice is a `rem:` slot whose box no longer exists, which would have
# marked the job failed.
#
# So the head was launched by hand with a 7800 MiB gate, and its queue state
# was pre-set to `running` to stop the dispatcher placing a second copy.
# Nothing then flips that state to `done`, and the eval behind it waits on
# exactly that. This does the flip, from the one fact that settles it: the
# final checkpoint on disk.
#
# Usage: q_adopt_head.sh <job id> <final checkpoint> [poll seconds]
set -uo pipefail

JOB="${1:?usage: q_adopt_head.sh <job id> <final ckpt> [poll]}"
CKPT="${2:?final checkpoint path}"
POLL="${3:-60}"
MIN_BYTES="${MIN_BYTES:-300000}"   # a head final is ~450 KB; see q_finish.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(dirname "$HERE")/results"
STATE="$RES/queue"
LOG="$RES/q_adopt_head.log"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [adopt $JOB] $*" | tee -a "$LOG"; }

log "watch $CKPT (poll ${POLL}s, min ${MIN_BYTES} B)"
while :; do
  if [ -f "$CKPT" ]; then
    sz=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$MIN_BYTES" ]; then
      # The trainer writes the final in one go, but wait one poll and
      # re-read the size anyway: a file caught mid-write that already
      # cleared the floor would hand the eval a truncated head.
      sleep 5
      sz2=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
      if [ "$sz" -eq "$sz2" ]; then
        echo done > "$STATE/$JOB.state"
        log "DONE — $sz B, state -> done; the dispatcher owns the eval now"
        exit 0
      fi
    fi
  fi
  # A head that died leaves no final and no process. Say so rather than
  # polling a checkpoint that will never arrive.
  if ! pgrep -f "train_forecasting_head.py.*$(basename "$(dirname "$CKPT")")" >/dev/null 2>&1; then
    [ -f "$CKPT" ] || { log "no trainer and no final — head died"; \
                        echo failed > "$STATE/$JOB.state"; exit 1; }
  fi
  sleep "$POLL"
done
