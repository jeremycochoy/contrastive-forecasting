#!/bin/bash
# Waits for score files of the elisa follower: rc 0 when one of the files (ANY=1) or all of them hold a value,
# rc 2 when the follower ends or a new failure line comes in a follow log of these runs, rc 3 when the polls are used.
# Usage: [ANY=1] wait_score_files.sh <follower pid> <score file> ...
PID=$1; shift
RES=${RES:-${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/ff/results}
PATTERN='FAILED|ABORT|Traceback|REFUSED|no score for|WAVE_END rc=[1-9]'
logs(){ for f in "$@"; do b=$(basename "$f" .txt); b=${b#score_}; echo "$RES/follow_${b%_bb*}.log"; done; }
count(){ local n=0 l; for l in $(logs "$@"); do [ -f "$l" ] && n=$((n + $(/usr/bin/grep -c -E "$PATTERN" "$l"))); done; echo $n; }
before=$(count "$@"); echo "$(date -u +%T) start: $# files, failure lines before: $before"
for poll in $(seq 1 115); do
  have=0; for f in "$@"; do [ -s "$f" ] && { have=$((have + 1)); echo "$(date -u +%T) poll $poll: $(basename "$f") $(head -c 8 "$f")"; }; done
  { [ "${ANY:-0}" = 1 ] && [ $have -ge 1 ]; } || [ $have -eq $# ] && exit 0
  [ "$(count "$@")" -gt "$before" ] && { echo "$(date -u +%T) poll $poll: FAILURE line"; exit 2; }
  kill -0 "$PID" 2>/dev/null || { echo "$(date -u +%T) poll $poll: FOLLOWER $PID ENDED"; exit 2; }
  sleep 60
done
echo "$(date -u +%T) poll budget used"; exit 3
