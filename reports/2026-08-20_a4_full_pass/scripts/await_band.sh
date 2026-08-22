#!/bin/bash
# #407 — block until one stop's band scores, then exit.
#
# This WAKES AN AGENT. It does no work of its own.
#
# Round 3's `await_redraw.sh` did both: it waited, and it ran the read-back
# on the way out. It ran as a harness background task, so it died with its
# session and the read-back never happened. The lesson is not "do not use a
# background task". The lesson is that no ARTEFACT may depend on one.
#
# So the read-back moved into `read_back.sh`, which `watchdog.sh` runs every
# hour and `replicate_heads.sh` runs the moment its band drains. Both of
# those outlive an agent. This script now carries no work at all, and it
# costs nothing when it dies.
#
# Usage: await_band.sh <stop_k> [seed ...]
#
# Exit codes:
#   0  every draw scored.
#   2  the deadline passed.
#   3  no chain for this stop is alive and the band is not scored.
#
# AWAIT_TIMEOUT  seconds before it gives up (default 25200, 7 h).
# AWAIT_POLL     seconds between checks (default 300).
set -uo pipefail

STOP_K="${1:?usage: await_band.sh <stop_k> [seed ...]}"
shift
SEEDS=("$@")
[ "${#SEEDS[@]}" -gt 0 ] || SEEDS=(20260723 20260724)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
export RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"

TIMEOUT="${AWAIT_TIMEOUT:-25200}"
POLL="${AWAIT_POLL:-300}"
LOG="$RES/await_${STOP_K}k.log"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-await${STOP_K}k] $*" | tee -a "$LOG"; }

score(){ cat "$PARENT_RES/score_A4_k3_bb${STOP_K}k_${1}_s${2}.txt" 2>/dev/null; }

scored(){
  local seed head
  for seed in "${SEEDS[@]}"; do
    for head in student teacher; do
      [ -s "$PARENT_RES/score_A4_k3_bb${STOP_K}k_${head}_s${seed}.txt" ] || return 1
    done
  done
  return 0
}

# A chain of this checkout's band script, at this stop.
SCRIPT="$(readlink -f "$HERE/replicate_heads.sh" 2>/dev/null)"
band_alive(){
  local p a1 a2 cwd full
  for p in $(pgrep -f 'replicate_heads\.sh' 2>/dev/null); do
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 2p)
    a2=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 3p)
    case "$a1" in */replicate_heads.sh) ;; *) continue;; esac
    [ "$a2" = "$(( STOP_K * 1000 ))" ] || continue
    case "$a1" in
      /*) full="$a1" ;;
      *)  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null) || continue
          [ -n "$cwd" ] || continue
          full="$cwd/$a1" ;;
    esac
    full=$(readlink -f "$full" 2>/dev/null)
    [ -n "$full" ] && [ "$full" = "$SCRIPT" ] && return 0
  done
  return 1
}

log "start seeds ${SEEDS[*]} timeout=${TIMEOUT}s poll=${POLL}s"
deadline=$(( $(date +%s) + TIMEOUT ))
last=""

while :; do
  if scored; then
    log "SCORED"
    for seed in "${SEEDS[@]}"; do
      log "  s${seed}  student $(score student "$seed")  teacher $(score teacher "$seed")"
    done
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    log "TIMEOUT after ${TIMEOUT}s"
    exit 2
  fi
  if ! band_alive; then
    log "the band at ${STOP_K}k is gone and not scored. band_queue.sh owns the retry."
    exit 3
  fi
  # One line per state change, so the log stays readable over hours.
  now=$(tail -1 "$RUNS/eval/A4_k3_bb${STOP_K}k_student_s${SEEDS[0]}/stop.log" \
        2>/dev/null | cut -c1-110)
  [ "$now" != "$last" ] && [ -n "$now" ] && { log "$now"; last="$now"; }
  sleep "$POLL"
done
