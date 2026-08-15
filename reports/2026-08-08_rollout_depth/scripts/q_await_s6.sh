#!/bin/bash
# #373 round 3 — session six's waiter.
#
# Blocks until every deliverable of the round has a score, or until the
# round budget runs out. It starts nothing and moves no job: the dispatcher
# owns the queue. This exists so the session waits on an event instead of
# polling in a loop.
#
# Exit codes tell the caller which one happened:
#   0  all 71 deliverables scored
#   1  the tick budget ran out (a heartbeat, not a failure)
#   2  credit fell below the floor the card set
#   3  the dispatcher died with work left
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
TICKS="${1:-55}"          # one tick a minute
FLOOR="${FLOOR:-5.00}"    # the card stops the job below $5

cd "$EXP" || exit 4

for (( t = 0; t < TICKS; t++ )); do
  line="$(python3 scripts/r2_coverage.py 2>/dev/null | grep -m1 '^deliverables')"
  done_n="$(sed -n 's/.*done \([0-9]*\).*/\1/p' <<<"$line")"
  want_n="$(sed -n 's/^deliverables \([0-9]*\).*/\1/p' <<<"$line")"

  if [ -n "$done_n" ] && [ -n "$want_n" ] && [ "$done_n" -ge "$want_n" ]; then
    echo "COMPLETE $done_n/$want_n after ${t}m"
    exit 0
  fi

  # The credit guard runs its own process and stops the box on its own. This
  # only needs to stop WAITING, so the session can post the blocking comment
  # the card asks for.
  cr="$(tail -1 "$RES/q_heartbeat.log" 2>/dev/null | sed -n 's/.*credit=\$\([0-9.]*\).*/\1/p')"
  if [ -n "$cr" ] && awk -v c="$cr" -v f="$FLOOR" 'BEGIN{exit !(c < f)}'; then
    echo "CREDIT $cr below floor $FLOOR — $done_n/$want_n"
    exit 2
  fi

  if ! kill -0 "$(cat "$RES/queue/dispatcher.pid" 2>/dev/null || echo 0)" 2>/dev/null; then
    echo "DISPATCHER GONE at $done_n/$want_n"
    exit 3
  fi

  sleep 60
done

echo "TICK $done_n/$want_n after ${TICKS}m"
exit 1
