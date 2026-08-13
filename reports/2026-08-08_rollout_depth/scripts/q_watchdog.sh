#!/bin/bash
# #373 round 3 — the session's event stream, and its hourly liveness probe.
#
# The queue's own daemons (dispatcher, supervisor, sync, guard, publisher)
# keep the work moving with no session attached. This script does NOT move
# work. It only says, on stdout, the things a session must react to:
#
#   1. every job state change             -> one line
#   2. every hour, a liveness probe       -> one line
#   3. a running backbone that stopped    -> one STALL line
#   4. the dispatcher dying               -> one line
#
# Rule 3 is the one that matters. A 2026-06-11 incident on another study
# left a rented machine degraded overnight: the process was alive, the log
# tailed clean, and the step counter had not moved for hours. Money drained
# at full rate. So the hourly probe compares a STEP COUNTER against the
# previous probe and says so when it has not advanced, rather than reporting
# that a process exists.
#
# Usage: BOX_ID=.. BOX_HOST=.. BOX_PORT=.. bash q_watchdog.sh [poll seconds]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
POLL="${1:-120}"
PROBE="${PROBE:-3600}"
BOX_ID="${BOX_ID:?BOX_ID}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)
rsh(){ timeout 90 ssh -n "${SSH_OPTS[@]}" -p "$BOX_PORT" "root@$BOX_HOST" "$@" 2>/dev/null; }

say(){ echo "$*"; echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$RES/q_watchdog.log"; }

ids(){ awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"; }
st(){ cat "$STATE/$1.state" 2>/dev/null || echo queued; }

# One number that must move between probes: every step line every training
# log holds, summed. It only grows, and it grows only when some process
# writes a step. awk does the sum — the rented container has no `bc`.
sum_steps(){ tr -dc '0-9\n' | awk '{n+=$1} END{printf "%.0f", n+0}'; }
progress(){
  local r l
  r="$(rsh "grep -hoE '^\[ *[0-9]+\]' /root/cf/reports/2026-08-08_rollout_depth/results/run_*.log /root/cf373_runs/eval/*/head.log 2>/dev/null" | sum_steps)"
  l="$(grep -hoE '^\[ *[0-9]+\]' "$RES"/run_*.log /home/jupyter/cf373_r3/sync/eval/*/head.log 2>/dev/null | sum_steps)"
  echo $(( ${r:-0} + ${l:-0} ))
}

declare -A prev
for id in $(ids); do prev[$id]="$(st "$id")"; done
say "[watch] armed: $(ids | wc -l) jobs, poll ${POLL}s, probe ${PROBE}s, box $BOX_ID"

last_probe=0; last_prog=0; elapsed=0
while :; do
  # ---------------------------------------------------------- state changes
  for id in $(ids); do
    now="$(st "$id")"
    if [ "$now" != "${prev[$id]}" ]; then
      say "[job] $id ${prev[$id]} -> $now"
      prev[$id]="$now"
    fi
  done

  # ------------------------------------------------------------- the counts
  nrun=0; ndone=0; nfail=0; nleft=0
  for id in $(ids); do case "$(st "$id")" in
    running) nrun=$(( nrun + 1 ));; done) ndone=$(( ndone + 1 ));;
    failed)  nfail=$(( nfail + 1 ));; *) nleft=$(( nleft + 1 ));; esac; done

  # ------------------------------------------------------- dispatcher alive
  dp="$(cat "$STATE/dispatcher.pid" 2>/dev/null | tr -dc '0-9')"
  if [ -n "$dp" ] && ! grep -qs 'q_run' "/proc/$dp/cmdline"; then
    say "[ALERT] dispatcher pid $dp is gone (supervisor should restart it within 5 min)"
  fi

  # ------------------------------------------------------- the hourly probe
  if [ "$elapsed" -ge "$last_probe" ]; then
    last_probe=$(( elapsed + PROBE ))
    credit="$(cd "$VDIR" && timeout 90 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')"
    spent="$(cd "$VDIR" && timeout 90 vastrun-status 2>/dev/null \
             | awk -v b="$BOX_ID" '$1==b{for(i=NF;i>0;i--) if($i ~ /^\$/){print $i; exit}}')"
    rgpu="$(rsh "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" | tr -d ' ' | paste -sd/ -)"
    lgpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd/ -)"
    prog="$(progress)"
    delta=$(( prog - last_prog ))
    if [ "$last_prog" -gt 0 ] && [ "$delta" -le 0 ] && [ "$nrun" -gt 0 ]; then
      say "[STALL] step counters flat over ${PROBE}s with $nrun job(s) running — box_gpu=${rgpu:-down} elisa_gpu=${lgpu:-?} credit=${credit:-?}"
    else
      say "[probe] credit=${credit:-?} box_spent=${spent:-?} box_gpu=${rgpu:-down} elisa_gpu=${lgpu:-?} run=$nrun done=$ndone fail=$nfail left=$nleft steps+=${delta}"
    fi
    last_prog="$prog"
    case "${credit:-}" in
      \$*) c="${credit#\$}"; awk -v c="$c" 'BEGIN{exit !(c < 7)}' \
             && say "[ALERT] credit ${credit} is near the \$5 floor" ;;
    esac
  fi

  # -------------------------------------------------------------- all done?
  if [ "$nleft" -eq 0 ] && [ "$nrun" -eq 0 ]; then
    say "[watch] queue drained: done=$ndone failed=$nfail"; exit 0
  fi

  sleep "$POLL"; elapsed=$(( elapsed + POLL ))
done
