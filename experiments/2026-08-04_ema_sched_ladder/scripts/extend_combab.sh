#!/bin/bash
# #393 — spend what is left of the ladder past bb100k on the cell the card
# names first: arm6_v2 combab, aligned on the student and on the teacher.
#
# Usage:  nohup setsid bash scripts/extend_combab.sh >> results/extend.log 2>&1 &
#
# The spend order has been "all ten cells to bb100k first, extensions with
# what is left" since PR #394, and `results/HOLD_ABOVE` is how it was
# enforced under drivers that were already running: ladder.py re-reads that
# file at every stop, so lowering it to 100000 parked every cell at the
# second rung with a `session_end` row.
#
# HOLD_ABOVE is global, and the drivers alive now read it. Raising it while
# they run would offer the extension to whichever cell reached its stop
# first, and the two arm6_v2 combab cells are not the fastest — they are the
# slowest, being the two that share elisa's GPUs with every head training on
# this machine. So this waits for the parked drivers to exit, THEN raises
# the ceiling, THEN starts a driver for each combab cell and nothing else.
#
# Relaunching a cell is the ordinary resume path and it is self-limiting.
# climb() replays the walk from step 0: run_leg.sh skips a leg whose target
# checkpoint exists, stop_scores skips a head ladder.csv already holds, and
# the extend rule is re-derived from the scores on disk. A cell the rule
# stopped at bb100k therefore exits in seconds without training anything.
# Nothing here decides to extend; it only removes the ceiling that was
# holding back a decision the rule makes.
#
# The two cells go on different GPUs. gpu_gate.sh serialises CUDA work per
# device, so sharing one would turn two 100k legs into one after the other.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="${WT:-$(dirname "$(dirname "$EXP")")}"   # the worktree the drivers run from
cd "$EXP" || exit 2

CEILING="${CEILING:-200000}"
POLL="${EXTEND_POLL:-300}"
DEADLINE_H="${EXTEND_DEADLINE_H:-8}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [extend] $*"; }

say "waiting for the parked drivers to exit before raising HOLD_ABOVE to $CEILING"
waited=0
while pgrep -f '[l]adder\.py --cells' >/dev/null 2>&1; do
  if [ "$waited" -ge $(( DEADLINE_H * 3600 )) ]; then
    say "ABORT: drivers still alive after ${DEADLINE_H}h — $(pgrep -af '[l]adder\.py --cells' | tr '\n' ';')"
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

# One extension leg is 100k steps at the 3.5-4.4 steps/s these two cells
# measure on elisa, so roughly 7 h, and then two 30k heads and two evals.
# Nothing here is rented, so the only budget it spends is wall clock.
launch(){  # <cell> <gpu>
  local cell="$1" gpu="$2"
  WT="$WT" RUNS=/home/jupyter/checkpoints_backup/cf-393 BB_GPU="$gpu" \
    nohup setsid python3 -u scripts/ladder.py --cells "$cell" \
    >> "results/ladder_${cell}.log" 2>&1 < /dev/null &
  say "launched $cell on GPU $gpu, log results/ladder_${cell}.log"
}

launch arm6_v2_combab_alignS 0
sleep 5
launch arm6_v2_combab_alignT 1
sleep 20
say "drivers now: $(pgrep -af '[l]adder\.py --cells' | sed 's/.*--cells //' | tr '\n' ' ')"
