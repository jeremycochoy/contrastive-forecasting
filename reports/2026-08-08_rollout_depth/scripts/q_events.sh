#!/bin/bash
# #373 round 3 — one stdout line per thing worth knowing, until the queue
# drains.
#
# q_await.sh blocks and returns once. This runs for the whole queue and
# emits an event per occurrence, so a session learns about each job as it
# lands rather than waking on a timer and re-reading state that has not
# moved.
#
# It emits, and nothing else:
#
#   JOB <id> <old> -> <new>      a job changed state
#   SCORE <tag> <value>          a GM-Relative MASE landed
#   BUDGET $<credit>             credit crossed the next threshold down
#   FLOOR $<credit> < $<floor>   the guard is about to stop everything
#   BOX DOWN                     three ssh misses, five minutes apart
#   IDLE <n> free slot(s)        cards idle with the queue not empty
#   DRAINED                      nothing queued and nothing running
#
# Silence means nothing changed. A crash is not silent: a job that dies
# writes `failed`, and that is a JOB line.
#
# Usage: BOX_HOST=.. BOX_PORT=.. bash q_events.sh [poll seconds]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
STATE="$RES/queue"
Q="$HERE/q_queue.tsv"
POLL="${1:-120}"
FLOOR="${FLOOR:-5.50}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"
export PATH="$HOME/.local/bin:$PATH"

ids(){ awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"; }
st(){ cat "$STATE/$1.state" 2>/dev/null || echo queued; }

declare -A was=()
for id in $(ids); do was[$id]="$(st "$id")"; done
declare -A seen_score=()
for f in "$RES"/score_*.txt; do [ -f "$f" ] && seen_score["$f"]=1; done

# Credit is reported only when it crosses one of these going down, so a
# quiet hour stays quiet and a spend that matters does not.
STEPS="20 15 10 7"
next_step(){ local c="$1" s
  for s in $STEPS; do awk -v a="$c" -v b="$s" 'BEGIN{exit !(a<b)}' && echo "$s"; done | head -1; }
reported=""

miss=0; t=0
while :; do
  sleep "$POLL"; t=$(( t + POLL ))

  for id in $(ids); do
    now="$(st "$id")"
    [ "$now" = "${was[$id]}" ] && continue
    echo "JOB $id ${was[$id]} -> $now"
    was[$id]="$now"
  done

  for f in "$RES"/score_*.txt; do
    [ -f "$f" ] || continue
    [ -n "${seen_score[$f]:-}" ] && continue
    seen_score["$f"]=1
    tag="$(basename "$f" .txt)"; tag="${tag#score_}"
    echo "SCORE $tag $(tr -d ' \n' < "$f")"
  done

  # Cards idle against a queue that is not empty is the failure this round
  # was built to avoid, so it is an event and not a silence.
  q_left=0; running=0
  for id in $(ids); do case "$(st "$id")" in
    queued) q_left=$(( q_left + 1 )) ;; running) running=$(( running + 1 )) ;; esac; done
  if [ "$q_left" -eq 0 ] && [ "$running" -eq 0 ]; then echo "DRAINED"; exit 0; fi

  # The box and the credit, every fifth poll.
  if [ $(( t % (POLL * 5) )) -eq 0 ]; then
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=25 -o BatchMode=yes -p "$BOX_PORT" \
           "root@$BOX_HOST" true 2>/dev/null; then
      [ "$miss" -ge 3 ] && echo "BOX BACK"
      miss=0
    else
      miss=$(( miss + 1 ))
      [ "$miss" -eq 3 ] && echo "BOX DOWN after 3 checks"
    fi
    c="$(timeout 120 vastrun-balance 2>/dev/null | awk '/Credit/{gsub(/\$/,"",$2); print $2}')"
    if [ -n "$c" ]; then
      if awk -v a="$c" -v b="$FLOOR" 'BEGIN{exit !(a<b)}'; then
        echo "FLOOR \$$c < \$$FLOOR"
      else
        s="$(next_step "$c")"
        if [ -n "$s" ] && [ "$s" != "$reported" ]; then
          echo "BUDGET \$$c (under \$$s)"; reported="$s"
        fi
      fi
    fi
  fi
done
