#!/bin/bash
# #373 round 3 — block until the queue does something worth reading, or the
# deadline passes. Written so a multi-hour run reports by event, not by poll.
#
# Usage: bash q_watch.sh [max seconds]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(dirname "$HERE")/results"; STATE="$RES/queue"; Q="$HERE/q_queue.tsv"
MAX="${1:-7200}"
n(){ grep -lx "$1" "$STATE"/*.state 2>/dev/null | wc -l; }
d0=$(n done); f0=$(n failed); t=0
while [ "$t" -lt "$MAX" ]; do
  sleep 120; t=$(( t + 120 ))
  [ "$(n done)" -ne "$d0" ] && { echo "EVENT: a job finished"; break; }
  [ "$(n failed)" -ne "$f0" ] && { echo "EVENT: a job failed"; break; }
  pgrep -f "bash scripts/q_run.sh" >/dev/null || { echo "EVENT: dispatcher gone"; break; }
done
echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') done=$(n done) failed=$(n failed) running=$(n running) ==="
tail -6 "$RES/q_run.log"
tail -2 "$RES/q_heartbeat.log" 2>/dev/null
