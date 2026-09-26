#!/bin/bash
# #412 — lane B, after the repeat seed: k3_r100_09_dec, and NOTHING after it.
#
# WHY THIS SCRIPT EXISTS. Lane B started as
# `ARMS="k3_r100_09 k3_r100_09b k3_r100_09_dec k32_r200_08"`. PR #413 then
# put `k32_r200_08` behind a gate: the climb of `k3_r100_09` to 100,000 steps
# says whether a 40,000-step stop can rank an 11.4M arm at all. A lane that
# starts `k32_r200_08` on its own spends 7.2 GPU-hours before that answer
# lands, and the gate then has nothing left to decide.
#
# The lane's arm list is a command line, and the loop that reads it already
# runs. So this script takes the lane over:
#
#   1. A human stops the lane's `phase1.sh` loop ALONE, with `kill <pid>`.
#      The leg under it keeps running: it is a child, not a member of a
#      signalled group, and its trainer, its AUC gate and its 20,000-step
#      saves are untouched.
#   2. This script waits for that leg to end.
#   3. It starts a new `phase1.sh` over `k3_r100_09b k3_r100_09_dec`.
#
# The new lane is idempotent, so step 3 costs nothing it does not owe: the
# repeat seed holds its 40,000-step checkpoint by then, its leg is a no-op,
# and the lane goes straight to that arm's head and its 97 GIFT-Eval configs.
#
# Usage:  lane_b_relay.sh <pid of the running run_arm.sh> [arm of that pid]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

PID="${1:?usage: lane_b_relay.sh <run_arm.sh pid> [arm]}"
ARM="${2:-k3_r100_09b}"
NEXT_ARMS="${CF412_RELAY_ARMS:-k3_r100_09b k3_r100_09_dec}"
GPU="${BB_GPU:-1}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 relay] $*" \
  | tee -a "$CF412_RESULTS/phase1.log"; }

log "waits for pid $PID (arm $ARM), then runs: $NEXT_ARMS on gpu $GPU"
log "k32_r200_08 stays out — PR #413 gates it on the 100,000-step climb"

# The command line, not the pid alone: a pid this long-lived script watches
# can be recycled by another program in the hours it waits.
while ps -p "$PID" -o cmd= 2>/dev/null | grep -q "run_arm.sh $ARM "; do
  sleep 60
done
log "pid $PID ended — arm $ARM holds $(cf412_bb_ckpt "$ARM" 40000 \
  | xargs -r basename)"

BB_GPU="$GPU" STOPS=40000 EVAL_SHARDS="${EVAL_SHARDS:-6}" ARMS="$NEXT_ARMS" \
  bash "$HERE/phase1.sh"
rc=$?
log "lane B done rc=$rc"
exit $rc
