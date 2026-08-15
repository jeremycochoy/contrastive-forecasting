#!/bin/bash
# #373 round 3 — one round's wait, for the session that watches the queue.
#
# The queue's daemons move the work with no session attached. This script
# moves nothing. It BLOCKS until the session has a reason to wake, then
# prints why and exits, so the harness re-invokes the session on the exit.
#
# It returns on the FIRST of:
#   1. the queue holds no job that is neither done nor failed   -> COMPLETE
#   2. credit falls under the floor                             -> CREDIT
#   3. a job fails                                              -> FAIL
#   4. the dispatcher dies                                      -> DISPATCHER
#   5. the round's wall clock runs out                          -> HOURLY
#
# Rule 5 is the hourly heartbeat: a session that waits only on notifications
# learns nothing when the thing that should notify it is the thing that died.
#
# Usage: bash q_await_round.sh [round seconds] [credit floor]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
ROUND="${1:-3600}"
FLOOR="${2:-5.00}"
POLL=60

jobs_left(){ local n=0 id s
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
    s="$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
    case "$s" in done|failed) ;; *) n=$(( n + 1 ));; esac
  done; echo "$n"; }

jobs_failed(){ grep -l '^failed' "$STATE"/*.state 2>/dev/null | wc -l; }

credit(){ timeout 60 vastai show user --raw 2>/dev/null \
  | python3 -c 'import sys,json
try: print(round(float(json.load(sys.stdin)["credit"]),2))
except Exception: print("")' 2>/dev/null; }

deadline=$(( SECONDS + ROUND ))
f0="$(jobs_failed)"

while :; do
  left="$(jobs_left)"
  [ "$left" -eq 0 ] && { echo "WAKE=COMPLETE left=0"; break; }

  f="$(jobs_failed)"
  if [ "$f" -gt "$f0" ]; then
    echo "WAKE=FAIL failed=$f left=$left"
    grep -l '^failed' "$STATE"/*.state 2>/dev/null | xargs -r -n1 basename | sed 's/.state$//' | tr '\n' ' '
    echo; break
  fi

  if ! pgrep -f 'q_run\.sh' >/dev/null 2>&1; then
    echo "WAKE=DISPATCHER dispatcher not running, left=$left"; break
  fi

  c="$(credit)"
  if [ -n "$c" ] && [ "$(python3 -c "print(1 if $c < $FLOOR else 0)")" = 1 ]; then
    echo "WAKE=CREDIT credit=\$$c floor=\$$FLOOR left=$left"; break
  fi

  [ "$SECONDS" -ge "$deadline" ] && { echo "WAKE=HOURLY left=$left credit=\$${c:-?}"; break; }
  sleep "$POLL"
done

# What the session prints first, every round.
echo "--- coverage $(date -u '+%Y-%m-%dT%H:%M:%SZ') ---"
timeout 300 python3 "$HERE/r2_coverage.py" 2>&1
echo "--- queue ---"
for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
  s="$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
  [ "$s" = done ] || echo "  $s	$id"
done
echo "--- credit \$$(credit) ---"
