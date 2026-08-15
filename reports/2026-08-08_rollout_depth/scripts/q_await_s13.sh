#!/bin/bash
# #373 round 3, session thirteen — block until the round moves enough to report.
#
# The round's tail is A4: backbone to 200k, then one student head, then one
# 97-config eval. That is about six hours, and nothing in it needs a hand.
# So this blocks, and returns only on something a session must say or act on:
#
#   every job terminal           the round is done             exit 0
#   DELTA jobs finished          enough moved to reprint       exit 6
#   a job failed                 the dispatcher needs a hand   exit 5
#   the box stops answering      three misses, 10 min apart    exit 3
#   credit under the floor       the guard is about to stop    exit 4
#   the deadline                 nothing moved, say so         exit 7
#
# It writes an hourly heartbeat to results/q_s13_heartbeat.log the whole time,
# so a dead session still leaves a record of where the round got to.
#
# Usage: bash q_await_s13.sh [max seconds] [poll seconds] [delta jobs]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
STATE="$RES/queue"
Q="$HERE/q_queue.tsv"
HB="$RES/q_s13_heartbeat.log"
MAX="${1:-9000}"
POLL="${2:-120}"
DELTA="${3:-4}"
FLOOR="${FLOOR:-5.50}"
BOX_HOST="${BOX_HOST:-ssh6.vast.ai}"
BOX_PORT="${BOX_PORT:-37390}"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"
export PATH="$HOME/.local/bin:$PATH"

ids(){ awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"; }
st(){ cat "$STATE/$1.state" 2>/dev/null || echo queued; }
ndone(){ local n=0; for id in $(ids); do [ "$(st "$id")" = done ] && n=$((n+1)); done; echo "$n"; }
counts(){ local d=0 r=0 q=0 f=0 s
          for id in $(ids); do s="$(st "$id")"
            case "$s" in done) d=$((d+1));; running) r=$((r+1));;
                         failed) f=$((f+1));; *) q=$((q+1));; esac; done
          echo "done=$d running=$r queued=$q failed=$f"; }
open_ids(){ local s; for id in $(ids); do s="$(st "$id")"
              case "$s" in done|failed) ;; *) echo "$id=$s";; esac; done; }
failed_ids(){ local s; for id in $(ids); do s="$(st "$id")"
                [ "$s" = failed ] && echo "$id"; done; }

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$HB"; }

D0="$(ndone)"
log "s13 await starts. $(counts) wake at done=$(( D0 + DELTA )) or terminal"

t=0; miss=0; lasthb=0; c=""
while [ "$t" -lt "$MAX" ]; do
  sleep "$POLL"; t=$(( t + POLL ))

  # Terminal states first: the round is over, or a job needs a hand.
  if [ -z "$(open_ids)" ]; then
    log "WOKE: every job terminal. $(counts)"; exit 0
  fi
  f="$(failed_ids)"
  if [ -n "$f" ]; then
    log "WOKE: FAILED $(echo $f | tr '\n' ' '). $(counts)"; exit 5
  fi

  # Enough finished to be worth a round report.
  n="$(ndone)"
  if [ "$(( n - D0 ))" -ge "$DELTA" ]; then
    log "WOKE: +$(( n - D0 )) jobs done. $(counts)"; exit 6
  fi

  # The box and the credit, every fifth poll.
  if [ $(( t % (POLL * 5) )) -eq 0 ]; then
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
           -o ConnectTimeout=25 -o BatchMode=yes -p "$BOX_PORT" \
           "root@$BOX_HOST" true 2>/dev/null; then
      miss=0
    else
      miss=$(( miss + 1 ))
      log "box unreachable ($miss)"
      [ "$miss" -ge 3 ] && { log "WOKE: BOX DOWN after $miss checks. $(counts)"; exit 3; }
    fi
    c="$(cd "$VDIR" && timeout 120 vastrun-balance 2>/dev/null \
         | awk '/Credit/{gsub(/\$/,"",$2); print $2}')"
    if [ -n "$c" ] && awk -v a="$c" -v b="$FLOOR" 'BEGIN{exit !(a<b)}'; then
      log "WOKE: credit \$$c under floor \$$FLOOR. $(counts)"; exit 4
    fi
  fi

  # Heartbeat, hourly, with the open jobs named.
  if [ $(( t - lasthb )) -ge 3600 ]; then
    lasthb="$t"
    log "hb +$((t/60))m $(counts) credit=\$${c:-?} open: $(open_ids | tr '\n' ' ')"
  fi
done
log "WOKE: deadline ${MAX}s. $(counts) open: $(open_ids | tr '\n' ' ')"
exit 7
