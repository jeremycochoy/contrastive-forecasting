#!/bin/bash
# #393 — spend what is left of the ladder past bb100k, in the card's order.
#
# Usage:  nohup setsid bash scripts/extend_queue.sh >> results/extend.log 2>&1 &
#         QUEUE="cell1 cell2" CEILING=200000 bash scripts/extend_queue.sh
#
# Replaces extend_combab.sh, which launched exactly two cells. The queue is
# the same idea with the two things that turned out to matter: an order, and
# a cap.
#
# ORDER. The spend order has been "all ten cells to bb100k first, extensions
# with what is left, arm6_v2 combab on student and on teacher leading" since
# PR #394. That pair leads because it leads both parent reports and gives the
# head-to-head. arm5_combab_alignT follows because it is the one cell whose
# rule has already said extend — `both_down` at bb100k, 1.2797 and 1.2772
# against 1.3334 and 1.3190 — and it was parked by the spend order rather
# than by the rule. It was rehomed onto elisa when its box went back.
#
# CAP. Two, one per GPU. gpu_gate.sh serialises CUDA work per device, so a
# third driver would not run a third leg, it would queue behind one of the
# first two and hold a 100k leg's worth of wall clock for nothing. The cap
# is enforced by counting drivers, so a cell that exits in seconds frees its
# slot immediately.
#
# HOLD_ABOVE is global and the parked drivers alive now read it at every
# stop. Raising it while they run would offer the extension to whichever
# cell reached its stop first, and the arm6_v2 combab pair are the slowest,
# sharing elisa's GPUs with every head training on this machine. So this
# waits for them to exit, THEN raises the ceiling, THEN starts the queue.
#
# Nothing here decides to extend. climb() replays a cell from step 0,
# skipping legs whose checkpoint exists and heads ladder.csv already holds,
# and re-derives the rule from the scores on disk. A cell the rule stopped
# exits in seconds having trained nothing. This only removes the ceiling
# that was holding back a decision the rule makes.
#
# Every cell in the queue is an elisa cell, so the extension does not touch
# the vast.ai credit.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="${WT:-$(dirname "$(dirname "$EXP")")}"
cd "$EXP" || exit 2

QUEUE="${QUEUE:-arm6_v2_combab_alignS arm6_v2_combab_alignT arm5_combab_alignT}"
CEILING="${CEILING:-200000}"
MAX_DRIVERS="${MAX_DRIVERS:-2}"
POLL="${EXTEND_POLL:-300}"
DEADLINE_H="${EXTEND_DEADLINE_H:-10}"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [extend] $*"; }
drivers(){ local n; n=$(pgrep -fc '[l]adder\.py --cells' 2>/dev/null); echo "${n:-0}"; }
running(){ pgrep -f "[l]adder\.py --cells.*$1" >/dev/null 2>&1; }

say "queue: $QUEUE  (ceiling $CEILING, at most $MAX_DRIVERS drivers)"
if [ -n "${SKIP_WAIT:-}" ]; then
  # The drain exists only so the ceiling is not raised under a driver that
  # would take the extension out of turn. A second queue behind the first
  # reads a ceiling that is ALREADY $CEILING, so there is nothing to raise
  # and nothing to take out of turn; waiting would just park the queue
  # behind a 100k leg it is meant to run alongside.
  say "SKIP_WAIT: ceiling is already $(cat results/HOLD_ABOVE 2>/dev/null), not draining"
else
  say "waiting for the parked drivers to exit before raising HOLD_ABOVE"
  waited=0
  while [ "$(drivers)" -gt 0 ]; do
    if [ "$waited" -ge $(( DEADLINE_H * 3600 )) ]; then
      say "ABORT: drivers still alive after ${DEADLINE_H}h — $(pgrep -af '[l]adder\.py --cells' | sed 's/.*--cells //' | tr '\n' ' ')"
      exit 1
    fi
    [ $(( waited % 1800 )) -eq 0 ] && [ "$waited" -gt 0 ] && \
      say "still waiting (${waited}s): $(pgrep -af '[l]adder\.py --cells' | sed 's/.*--cells //' | tr '\n' ' ')"
    sleep "$POLL"
    waited=$(( waited + POLL ))
  done
  say "no driver left after ${waited}s"

  echo "$CEILING" > results/HOLD_ABOVE.tmp && mv -f results/HOLD_ABOVE.tmp results/HOLD_ABOVE
  say "HOLD_ABOVE = $(cat results/HOLD_ABOVE)"
fi

# Which card to hand the next cell. Round-robin from 0 was right when the
# queue started on an empty machine; a queue that starts while GPU 0 is
# already carrying a leg would put its first cell there and leave the other
# card idle. Ask the driver instead: the GPU with no CUDA process on it.
free_gpu() {
  local g busy
  busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
          | sort -u | wc -l)"
  for g in 0 1; do
    local uuid n
    uuid="$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader 2>/dev/null)"
    n="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
         | grep -cF "$uuid")"
    [ "${n:-1}" -eq 0 ] && { printf '%s' "$g"; return 0; }
  done
  printf '%s' "${1:-0}"   # both busy: fall back to the caller's guess
}

gpu=0
for cell in $QUEUE; do
  # A cell nobody rehomed would be refused by run_leg.sh's claim guard
  # before it burned anything, but the log line is easier to read here.
  owner="$(awk -v c="$cell" '$1==c {print $2}' results/cell_claims.txt | head -1)"
  me="$(tr -dc 'a-zA-Z0-9_-' <results/MACHINE 2>/dev/null)"
  if [ -n "$owner" ] && [ "$owner" != "$me" ]; then
    say "SKIP $cell — claimed by '$owner', not '$me'; rehome_cell.sh first"
    continue
  fi
  if running "$cell"; then say "SKIP $cell — already has a driver"; continue; fi

  while [ "$(drivers)" -ge "$MAX_DRIVERS" ]; do
    say "$cell waits: $(drivers) driver(s) up, cap $MAX_DRIVERS"
    sleep "$POLL"
  done

  gpu="$(free_gpu "$gpu")"
  WT="$WT" RUNS="$RUNS" BB_GPU="$gpu" \
    nohup setsid python3 -u scripts/ladder.py --cells "$cell" \
    >> "results/ladder_${cell}.log" 2>&1 < /dev/null &
  say "launched $cell on GPU $gpu, log results/ladder_${cell}.log"
  gpu=$(( (gpu + 1) % 2 ))
  sleep 30
done

say "queue dispatched; drivers now: $(pgrep -af '[l]adder\.py --cells' | sed 's/.*--cells //' | tr '\n' ' ')"
