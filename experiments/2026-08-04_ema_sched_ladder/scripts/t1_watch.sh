#!/bin/bash
# #393 — event stream for the tier-1 bb40k pool, one line per state change.
#
# Usage: bash scripts/t1_watch.sh
#
# Emits a line when a job scores, when a job fails, when the pool loses its
# last worker, and once an hour regardless so a silent stream is
# distinguishable from a dead one. Exits 0 when all 24 tier-1 jobs are
# scored, which is the signal to stop and release the GPUs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
export WT="${WT:-$(dirname "$(dirname "$EXP")")}"
export RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
export CF393_BB40K_TIERS=t1

DRIVER_LOG="$EXP/results/seed_replicates_bb40k.log"
POLL="${T1_WATCH_POLL:-300}"
HEARTBEAT_EVERY=$(( 3600 / POLL ))

status(){ bash "$HERE/seed_replicates_bb40k.sh" --status 2>/dev/null; }
scored_lines(){ awk '$6 ~ /^[0-9]/ {print $1" "$3" s"$4" = "$6}' | sort; }

prev="$(status | scored_lines)"
prev_fail="$(grep -c '^\[.*FAIL ' "$DRIVER_LOG" 2>/dev/null || echo 0)"
n=0
echo "t1 watch armed — $(status | tail -1)"

while :; do
  sleep "$POLL"
  n=$(( n + 1 ))
  st="$(status)"
  cur="$(printf '%s\n' "$st" | scored_lines)"

  comm -13 <(printf '%s\n' "$prev") <(printf '%s\n' "$cur") \
    | while read -r l; do [ -n "$l" ] && echo "SCORED $l"; done
  prev="$cur"

  # Failures release their claim, so they are only visible in the driver log.
  fail="$(grep -c '^\[.*FAIL ' "$DRIVER_LOG" 2>/dev/null || echo 0)"
  if [ "${fail:-0}" -gt "${prev_fail:-0}" ]; then
    grep '^\[.*FAIL ' "$DRIVER_LOG" | tail -n $(( fail - prev_fail ))
    prev_fail="$fail"
  fi

  tally="$(printf '%s\n' "$st" | tail -1)"
  done_n="$(printf '%s' "$tally" | sed -n 's#^\([0-9]*\)/.*#\1#p')"

  # No worker and no supervisor means nothing will finish the rest.
  if ! pgrep -f 'seed_replicates_bb40k.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'bb40k_supervisor.sh' >/dev/null 2>&1; then
    echo "POOL EMPTY — no driver and no supervisor alive. $tally"
    exit 3
  fi

  if [ "${done_n:-0}" -ge 24 ]; then
    echo "T1 COMPLETE — $tally"
    exit 0
  fi

  [ $(( n % HEARTBEAT_EVERY )) -eq 0 ] && echo "heartbeat — $tally"
done
