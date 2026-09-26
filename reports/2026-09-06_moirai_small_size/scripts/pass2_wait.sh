#!/bin/bash
# #412 pass 2 — block until the card's state changes, then print it and exit.
#
# The agent session waits on THIS, not on a poll loop. It returns on the first
# event that changes an answer:
#
#   * a new score file, which is one arm at one stop
#   * a collapse note, which is the AUC gate's verdict on an arm
#   * a lane that ended, which is every leg of that card
#
# It also returns on CF412_WAIT_MAX seconds, so a stalled box still reports.
#
# Usage:  bash scripts/pass2_wait.sh
#         CF412_WAIT_MAX=7200 bash scripts/pass2_wait.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

POLL="${CF412_WAIT_POLL:-120}"
MAX="${CF412_WAIT_MAX:-10800}"

# The state this waiter watches: every written score, every collapse note, and
# whether each lane still runs. A lane is a `pass2_lane.sh` whose EXECUTABLE is
# bash, never a watcher shell that merely names the file (`head_busy.sh`).
state(){
  local arm stop
  for arm in $CF412_ARMS; do
    for stop in $CF412_STOPS; do
      [ -s "$(cf412_score_file "$arm" "$stop")" ] && printf 'score %s %s\n' "$arm" "$stop"
    done
    [ -f "$(cf412_collapse_file "$arm")" ] && printf 'collapsed %s\n' "$arm"
  done
  printf 'lanes %s\n' "$(ps -eo args --no-headers 2>/dev/null \
    | awk '$1 ~ /bash$/ && $2 ~ /pass2_lane\.sh$/ { n++ } END { print n+0 }')"
}

before="$(state)"
waited=0
while [ "$waited" -lt "$MAX" ]; do
  sleep "$POLL"; waited=$(( waited + POLL ))
  now="$(state)"
  [ "$now" = "$before" ] || break
done

echo "===== pass 2 wait returned after ${waited}s ====="
diff <(printf '%s\n' "$before") <(printf '%s\n' "$now") | sed 's/^/  /'
echo
bash "$HERE/watch_status.sh"
