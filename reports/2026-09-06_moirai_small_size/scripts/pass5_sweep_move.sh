#!/bin/bash
# #412 — move the head sweep to the card that clears the head gate.
#
# WHY. `CF412_HEAD_VRAM_MIB` is 9,000. Two Jupyter kernels took 11,692 MiB on
# each card at 19:26, and after that only ONE card can clear that gate:
#
#   GPU 0   2,818 MiB free + 6,472 that the pass-4 leg gives back = 9,290
#   GPU 1     542 MiB free + 6,472 = 7,014
#
# So a head that starts on GPU 1 waits on its `flock` for the four hours of
# HEAD_VRAM_TIMEOUT and then aborts, and every later head of the card waits
# behind it. The sweep belongs on GPU 0 while the kernels hold the box.
#
# WHEN. Not now. The pass-4 lane trains its own head on GPU 0 inline, and two
# head trains on one card is what the `flock` exists to prevent. This script
# waits for that head to appear and then to END, and it moves the sweep in the
# window between it and the next leg.
#
# It restarts the sweep ONLY when no head trainer runs anywhere, so it never
# kills a loop that holds a head. A CPU evaluation is not a head trainer and
# does not block the move: it survives the restart, because `head_eval.sh`
# runs under `nohup` and a SIGTERM to the loop does not reach it.
#
# Usage:  nohup bash scripts/pass5_sweep_move.sh >>results/pass5_sweep_move.out 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

TARGET="${CF412_SWEEP_GPU:-0}"
TIMEOUT="${CF412_MOVE_TIMEOUT:-28800}"
POLL="${CF412_MOVE_POLL:-60}"
SWEEP="${CF412_SWEEP_SCRIPT:-$CF412_STUDY/run_snapshot/head_sweep.sh}"
mkdir -p "$CF412_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 sweep-move] $*" \
  | tee -a "$CF412_RESULTS/pass5_sweep_move.log"; }

# A head TRAINER, which holds the card. The CPU evaluation that follows it is
# a different process and holds no card.
head_pids(){ ps -eo pid,args --no-headers 2>/dev/null \
  | awk '$2 ~ /python/ && /train_forecasting_head/ { print $1 }'; }

sweep_pid(){ ps -eo pid,args --no-headers 2>/dev/null \
  | awk '$2 ~ /bash$/ && $3 ~ /head_sweep\.sh$/ { print $1 }' | head -1; }

log "waits for the pass-4 inline head on gpu $TARGET, then moves the sweep there"
waited=0
# Phase 1 — the inline head appears.
while [ -z "$(head_pids)" ] && [ "$waited" -lt "$TIMEOUT" ]; do
  sleep "$POLL"; waited=$(( waited + POLL ))
done
if [ -z "$(head_pids)" ]; then
  log "TIMEOUT after ${waited}s — no head train ever started. The sweep stays" \
      "where it is, and a session can move it by hand."
  exit 1
fi
log "a head train is up (pid $(head_pids | tr '\n' ' ')) — waits for it to end"

# Phase 2 — it ends. Then the card is free of head work.
while [ -n "$(head_pids)" ] && [ "$waited" -lt "$TIMEOUT" ]; do
  sleep "$POLL"; waited=$(( waited + POLL ))
done
if [ -n "$(head_pids)" ]; then
  log "TIMEOUT after ${waited}s — the head train never ended. No move."
  exit 1
fi

old="$(sweep_pid)"
if [ -n "$old" ]; then
  log "stops the sweep (pid $old)"
  pkill -P "$old" 2>/dev/null
  kill "$old" 2>/dev/null
  sleep 2
fi
BB_GPU="$TARGET" CF412_SWEEP_EVERY="${CF412_SWEEP_EVERY:-600}" \
  nohup setsid bash "$SWEEP" \
  >>"$CF412_RESULTS/head_sweep_gpu${TARGET}.out" 2>&1 &
sleep 10
new="$(sweep_pid)"
if [ -n "$new" ]; then
  log "the sweep runs on gpu $TARGET (pid $new," \
      "BB_GPU=$(tr '\0' '\n' <"/proc/$new/environ" 2>/dev/null | sed -n 's/^BB_GPU=//p'))"
else
  log "FAILED — the sweep did not come back up. Start it by hand:" \
      "BB_GPU=$TARGET nohup bash $SWEEP &"
  exit 1
fi
