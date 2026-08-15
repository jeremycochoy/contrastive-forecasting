#!/bin/bash
# #373 round 3 — block until the round's last deliverable is scored.
# The last number is A4 bb200k student. Exits when its score file holds a
# value AND the queue holds no job that is neither done nor failed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(cd "$HERE/../results" && pwd)"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
TARGET="$RES/score_A4_k3_bb200k_student.txt"
left(){ local n=0 id s
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
    s="$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
    case "$s" in done|failed) ;; *) n=$((n+1));; esac
  done; echo "$n"; }
DEADLINE=$(( SECONDS + 14400 ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if [ -s "$TARGET" ] && [ "$(left)" -eq 0 ]; then
    echo "=== LAST SCORE IN $(date -u +%H:%M:%SZ) ==="
    echo "A4_k3_bb200k_student = $(cat "$TARGET")"
    echo "--- queue ---"; echo "open jobs: $(left)"
    exit 0
  fi
  sleep 180
done
echo "=== DEADLINE $(date -u +%H:%M:%SZ): open=$(left) target=$( [ -s "$TARGET" ] && cat "$TARGET" || echo MISSING) ==="
tail -6 "$RES/q_run.log"
