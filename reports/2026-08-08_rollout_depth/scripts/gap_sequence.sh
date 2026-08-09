#!/bin/bash
# #373 — the review gaps, in the reviewer's order, on elisa's two 4090s.
#
#   1. gap 1's head is already training when this starts. Wait for it.
#   2. gap 4's step-time probe, on a GPU 0 with nothing of this study on it.
#      It has to come before the backbones: it is the one measurement whose
#      answer depends on the card being quiet, and the queue below keeps
#      that card busy for the rest of the day.
#   3. the backbone queue — gaps 2, 3, 5, 6 — two at a time.
#
# Usage: bash gap_sequence.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$WT/reports/2026-08-08_rollout_depth/results"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
G1_HEAD="$CF373_ROOT/eval/G1_B5pub_bb40k_student/qhead_G1_B5pub_bb40k_student_s20260722_final.pth"
mkdir -p "$RES"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [seq] $*" | tee -a "$RES/gaps_driver.log"; }

# 1. gap 1's head. Its GIFT-Eval runs on the cores afterwards and does not
#    hold the card, so the probe only waits for the head itself.
waited=0
while [ ! -f "$G1_HEAD" ]; do
  if [ "$waited" -ge 7200 ]; then
    log "gap 1 head did not appear in ${waited}s — going on to the probe anyway"
    break
  fi
  [ $(( waited % 600 )) -eq 0 ] && log "waiting for gap 1's head (${waited}s)"
  sleep 30; waited=$(( waited + 30 ))
done
[ -f "$G1_HEAD" ] && log "gap 1 head done after ${waited}s"
sleep 60      # let the trainer's context tear down before reading the card

# 2. gap 4. B5 only: it is the cell whose cost claim the report makes, and
#    the A3 pair is settled from the production logs (steptime_solo.csv).
log "probe START (B5, 600 steps, 3 reps, alternating, GPU 0)"
PROBE_TAG=_solo BB_GPU=0 bash "$HERE/steptime.sh" B5 600 3 \
  >>"$RES/gaps_driver.log" 2>&1
log "probe END rc=$?"

# 3. the backbone queue. Two workers on GPU 0: a d_model=64 model at batch
#    64 leaves the card about half idle, and the heads that follow each
#    backbone take whichever card has room.
log "queue START (2 workers, GPU 0)"
BB_GPU=0 bash "$HERE/gap_worker.sh" 0 2 >>"$RES/gap_worker0.log" 2>&1 &
w0=$!
sleep 20
BB_GPU=0 bash "$HERE/gap_worker.sh" 1 2 >>"$RES/gap_worker1.log" 2>&1 &
w1=$!
wait "$w0" "$w1"
log "queue drained"
