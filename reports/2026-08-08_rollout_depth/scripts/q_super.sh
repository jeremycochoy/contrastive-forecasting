#!/bin/bash
# #373 round 3 — keep one dispatcher alive, and never two.
#
# q_run.sh is the only thing that starts work on the four cards. It runs
# detached, so a dead session does not kill it, but nothing brought it back
# if it died on its own: the cards would then finish what they hold and go
# idle with a full queue. Two sessions already ended inside a minute, and
# the queue outlived them only because it was already detached.
#
# This loop checks every 5 minutes and restarts the dispatcher when it is
# gone. It refuses to start a second one: q_run.sh adopts running jobs by
# their own marker files, so one restart is safe and two concurrent
# dispatchers are not — round 2 put two processes on one run name that way.
#
# It stands down for the two states where a stopped dispatcher is correct:
# the budget guard wrote BLOCKED_BUDGET, or the queue drained.
#
# Usage: BOX_ID=.. BOX_HOST=.. BOX_PORT=.. bash q_super.sh [poll seconds]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
STATE="$RES/queue"
Q="$HERE/q_queue.tsv"
POLL="${1:-300}"
BOX_ID="${BOX_ID:?BOX_ID}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [super] $*" | tee -a "$RES/q_super.log"; }

# The dispatcher forks a subshell per local job, and that child carries the
# same argv. Counting by argv alone would read a running local head as a
# live dispatcher. The parent is the process whose own parent is init.
dispatcher_pid(){
  local p
  for p in $(pgrep -f 'bash scripts/q_run.sh' 2>/dev/null); do
    [ "$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)" = 1 ] && { echo "$p"; return 0; }
  done
  return 1
}

queue_left(){ local n=0 id s
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
    s="$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
    case "$s" in queued|running) n=$(( n + 1 )) ;; esac
  done; echo "$n"; }

log "watching dispatcher, poll ${POLL}s, box $BOX_ID"
while :; do
  sleep "$POLL"
  [ -f "$RES/BLOCKED_BUDGET" ] && { log "BLOCKED_BUDGET present — standing down"; exit 0; }
  [ "$(queue_left)" -eq 0 ] && { log "queue drained — standing down"; exit 0; }
  dispatcher_pid >/dev/null && continue
  log "dispatcher gone with $(queue_left) job(s) left — restarting"
  ( cd "$STUDY" && BOX_ID="$BOX_ID" BOX_HOST="$BOX_HOST" BOX_PORT="$BOX_PORT" \
      CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
      setsid nohup bash scripts/q_run.sh >> results/q_run.out 2>&1 < /dev/null & )
  sleep 30
  if dispatcher_pid >/dev/null; then log "restarted, pid $(dispatcher_pid)"
  else log "RESTART FAILED — see results/q_run.out"; fi
done
