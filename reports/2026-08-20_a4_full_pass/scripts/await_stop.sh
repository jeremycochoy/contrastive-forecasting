#!/bin/bash
# #407 — block until the DRIVER scores one stop, then exit.
#
# This WAKES AN AGENT. It does no work of its own.
#
# `await_band.sh` waits on a BAND: score files that carry `_s<seed>`. This
# one waits on the driver's own two numbers at one stop, which carry no
# seed suffix. Nothing else differs, and the same doctrine holds: no
# ARTEFACT may depend on this script. `read_back.sh` runs from the watchdog
# every tick and from `replicate_heads.sh` when a band drains, so the
# numbers reach the checkout whether this script lives or dies.
#
# It does NOT own a retry. The watchdog re-fires a dead driver, and it needs
# two quiet ticks to decide. So a driver that disappears is not an error
# here while the watchdog is up.
#
# Usage: await_stop.sh <stop_k> [head ...]
#
# Exit codes:
#   0  every head scored.
#   2  the deadline passed.
#   3  the driver is gone, the watchdog is gone, and the stop is not scored.
#
# AWAIT_TIMEOUT    seconds before it gives up (default 36000, 10 h).
# AWAIT_POLL       seconds between checks (default 120).
# AWAIT_HEARTBEAT  seconds between progress lines (default 1800).
set -uo pipefail

STOP_K="${1:?usage: await_stop.sh <stop_k> [head ...]}"
shift
HEADS=("$@")
[ "${#HEADS[@]}" -gt 0 ] || HEADS=(student teacher)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"
TRAIN_LOG="$PARENT_RES/run_cf393_arm6_v2_combab_alignS_cf373k3.log"

TIMEOUT="${AWAIT_TIMEOUT:-36000}"
POLL="${AWAIT_POLL:-120}"
HEARTBEAT="${AWAIT_HEARTBEAT:-1800}"
LOG="$RES/await_stop_${STOP_K}k.log"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-await-stop${STOP_K}k] $*" \
        | tee -a "$LOG"; }

score_file(){ echo "$PARENT_RES/score_A4_k3_bb${STOP_K}k_${1}.txt"; }

scored(){
  local head
  for head in "${HEADS[@]}"; do
    [ -s "$(score_file "$head")" ] || return 1
  done
  return 0
}

# The same `argv[1]` guard the watchdog uses. `pgrep` alone also matches the
# shell that launched the process and any tail that watches it, so this
# reads `argv[1]` out of `/proc`. It then resolves that path against the
# process's own working directory and demands THIS checkout's copy. Both
# the driver and the watchdog launch by a relative path, so a basename test
# would let a second worktree of the repo read as this study's.
proc_alive(){ # <basename>
  local p a1 cwd full want
  want=$(readlink -f "$HERE/$1" 2>/dev/null) || return 1
  for p in $(pgrep -f "$1" 2>/dev/null); do
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 2p)
    case "$a1" in */"$1") ;; *) continue ;; esac
    case "$a1" in
      /*) full="$a1" ;;
      *)  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null) || continue
          [ -n "$cwd" ] || continue
          full="$cwd/$a1" ;;
    esac
    full=$(readlink -f "$full" 2>/dev/null)
    [ -n "$full" ] && [ "$full" = "$want" ] && return 0
  done
  return 1
}

last_step(){
  grep -oE '^\[ *[0-9]+\]' "$TRAIN_LOG" 2>/dev/null | tail -1 | tr -dc '0-9'
}

log "start heads ${HEADS[*]} timeout=${TIMEOUT}s poll=${POLL}s heartbeat=${HEARTBEAT}s"
deadline=$(( $(date +%s) + TIMEOUT ))
next_beat=0

while :; do
  if scored; then
    log "SCORED"
    for head in "${HEADS[@]}"; do
      log "  $head  $(cat "$(score_file "$head")" 2>/dev/null | tr -d '\n')"
    done
    exit 0
  fi

  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    log "TIMEOUT after ${TIMEOUT}s"
    exit 2
  fi

  driver=up; proc_alive 'run_pass.sh' || driver=down
  watch=up;  proc_alive 'watchdog.sh' || watch=down

  if [ "$driver" = down ] && [ "$watch" = down ]; then
    log "the driver and the watchdog are both gone, and ${STOP_K}k is not scored"
    exit 3
  fi

  if [ "$now" -ge "$next_beat" ]; then
    have=""
    for head in "${HEADS[@]}"; do
      [ -s "$(score_file "$head")" ] && have="$have $head"
    done
    log "driver=$driver watchdog=$watch step=$(last_step) scored='${have# }'"
    next_beat=$(( now + HEARTBEAT ))
  fi

  sleep "$POLL"
done
