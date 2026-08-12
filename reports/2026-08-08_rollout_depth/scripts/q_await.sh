#!/bin/bash
# #373 round 3 — block until something happens worth reading, then return.
#
# The queue runs for about a day. Polling it in a loop from the session costs
# a turn per look and reads nothing new most of the time. This blocks instead,
# and returns on the first of:
#
#   a job finishes or fails      the queue moved
#   the box stops answering      three misses, 5 min apart
#   credit under the floor       the guard is about to stop everything
#   the deadline                 nothing happened, say so
#
# It prints what woke it and the queue's counts, and exits:
#   0 a job moved   3 box down   4 credit floor   7 deadline, no change
#
# Usage: bash q_await.sh [max seconds] [poll seconds]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
STATE="$RES/queue"
Q="$HERE/q_queue.tsv"
MAX="${1:-7200}"
POLL="${2:-120}"
FLOOR="${FLOOR:-6.00}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"
export PATH="$HOME/.local/bin:$PATH"

ids(){ awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"; }
snap(){ local id; for id in $(ids); do
          printf '%s=%s\n' "$id" "$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
        done; }
counts(){ local d=0 r=0 q=0 f=0 s
          for id in $(ids); do s="$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
            case "$s" in done) d=$((d+1));; running) r=$((r+1));;
                         failed) f=$((f+1));; *) q=$((q+1));; esac; done
          echo "done=$d running=$r queued=$q failed=$f"; }

before="$(snap)"
t=0; miss=0
while [ "$t" -lt "$MAX" ]; do
  sleep "$POLL"; t=$(( t + POLL ))

  now="$(snap)"
  if [ "$now" != "$before" ]; then
    echo "WOKE: queue moved after ${t}s"
    diff <(echo "$before") <(echo "$now") | grep '^>' | sed 's/^> /  /'
    counts; exit 0
  fi

  # The box, every fifth poll. Three misses in a row is down, not a blip.
  if [ $(( t % (POLL * 5) )) -eq 0 ]; then
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=25 -o BatchMode=yes -p "$BOX_PORT" \
           "root@$BOX_HOST" true 2>/dev/null; then
      miss=0
    else
      miss=$(( miss + 1 ))
      echo "[$(date -u '+%H:%M:%SZ')] box unreachable ($miss)"
      [ "$miss" -ge 3 ] && { echo "WOKE: BOX DOWN after $miss checks"; counts; exit 3; }
    fi
    c="$(cd "$VDIR" && timeout 120 vastrun-balance 2>/dev/null \
         | awk '/Credit/{gsub(/\$/,"",$2); print $2}')"
    if [ -n "$c" ] && awk -v a="$c" -v b="$FLOOR" 'BEGIN{exit !(a<b)}'; then
      echo "WOKE: credit \$$c under floor \$$FLOOR"; counts; exit 4
    fi
  fi
done
echo "WOKE: deadline ${MAX}s, queue unchanged"
counts
exit 7
