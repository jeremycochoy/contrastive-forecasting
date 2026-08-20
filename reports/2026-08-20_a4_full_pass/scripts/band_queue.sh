#!/bin/bash
# #407 round-3 gaps 2 and 7 — the two draws card 1 still owes.
#
# The watchdog fires the band at the LAST stop. Two more bands belong to
# this card, and neither one fits the watchdog's rule:
#
#   gap 2, the protocol re-draw at 200k. Head seed 20260722, the card's own
#          seed, on this machine and on this code. The band that runs now
#          holds 1.0660 from #373 (another round, another box) beside two
#          seeds drawn here, so head seed, machine and code version move
#          together. This draw holds the seed still and moves only the
#          machine and the code. It also answers a prior question: does
#          1.0660 reproduce here at all?
#   gap 7, the band at 450k. The interior stops carry one draw each, so a
#          move between 300k and 450k has no scale beside it.
#
# Both wait on card 1 rather than on a clock. Stage 1 waits for the seeds
# 20260723 and 20260724 chains to drain, because `head_vram_gate` serialises
# on one flock and a third chain would only queue behind them. Stage 2 waits
# for the 450,000-step checkpoint to land.
#
# This takes no GPU time from the driver: `BAND_GPU` is the card the driver
# is not on, and the tag of every draw carries its seed, so no draw can
# overwrite the card's own six numbers.
#
# Usage:
#   WT=<checkout> RUNS=<durable root> BAND_GPU=1 \
#     nohup setsid bash band_queue.sh > band_queue.out 2>&1 &
#
# QUEUE_PERIOD  seconds between checks (default 300).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
export RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"
export BAND_GPU="${BAND_GPU:-1}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"

PERIOD="${QUEUE_PERIOD:-300}"
REDRAW_SEED="${REDRAW_SEED:-20260722}"
REDRAW_STOP="${REDRAW_STOP:-200000}"
MID_STOP="${MID_STOP:-450000}"
LOG="$RES/band_queue.log"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-queue] $*" | tee -a "$LOG"; }

# Same `argv[1]` test the watchdog uses. `pgrep -f replicate_heads.sh` alone
# also matches the shell that launched it and any tail that watches it, so
# the test reads the command line out of `/proc`.
replicate_alive(){ # <stop>
  local p a1 a2
  for p in $(pgrep -f 'replicate_heads\.sh' 2>/dev/null); do
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 2p)
    a2=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 3p)
    case "$a1" in */replicate_heads.sh)
      [ "$a2" = "$1" ] && return 0 ;;
    esac
  done
  return 1
}

# Both heads of one (stop, seed) are scored.
seed_drained(){ # <stop> <seed>
  local k=$(( $1 / 1000 )) head
  for head in student teacher; do
    [ -s "$PARENT_RES/score_A4_k3_bb${k}k_${head}_s${2}.txt" ] || return 1
  done
  return 0
}

fire(){ # <stop> <seed...>
  local stop="$1"; shift
  local k=$(( stop / 1000 ))
  log "FIRE band at ${k}k, seeds $*, on GPU $BAND_GPU"
  WT="$WT" CF373_ROOT="$RUNS" BB_GPU="$BAND_GPU" \
    HEAD_VRAM_MIB="${BAND_VRAM_MIB:-6400}" \
    HEAD_VRAM_TIMEOUT="${BAND_VRAM_TIMEOUT:-28800}" \
    nohup setsid bash "$HERE/replicate_heads.sh" "$stop" "$@" \
      >>"$RES/replicate_${k}k.out" 2>&1 &
}

log "start period=${PERIOD}s gpu=$BAND_GPU redraw=${REDRAW_SEED}@${REDRAW_STOP} mid=${MID_STOP}"

stage1=pending
stage2=pending
seed_drained "$REDRAW_STOP" "$REDRAW_SEED" && stage1=done
[ -s "$RES/replicate_$(( MID_STOP / 1000 ))k.log" ] && stage2=fired

while :; do
  # Stage 1: the protocol re-draw at 200k, once card 1 is free.
  if [ "$stage1" = pending ]; then
    if seed_drained "$REDRAW_STOP" "$REDRAW_SEED"; then
      log "re-draw at $(( REDRAW_STOP / 1000 ))k is scored on both heads"
      stage1=done
    elif replicate_alive "$REDRAW_STOP"; then
      : # a band at this stop is up. Either the 23/24 pair or this one.
    else
      fire "$REDRAW_STOP" "$REDRAW_SEED"
      stage1=fired
    fi
  elif [ "$stage1" = fired ]; then
    if seed_drained "$REDRAW_STOP" "$REDRAW_SEED"; then
      log "re-draw DONE: student $(cat "$PARENT_RES/score_A4_k3_bb200k_student_s${REDRAW_SEED}.txt" 2>/dev/null)  teacher $(cat "$PARENT_RES/score_A4_k3_bb200k_teacher_s${REDRAW_SEED}.txt" 2>/dev/null)"
      stage1=done
    elif ! replicate_alive "$REDRAW_STOP"; then
      log "WARN: the re-draw is gone and not scored. Firing it again."
      stage1=pending
    fi
  fi

  # Stage 2: the band at 450k, once that checkpoint lands. It fires on the
  # CHECKPOINT, so the two extra seeds train while the driver still scores
  # the protocol seed at the same stop.
  if [ "$stage2" = pending ] && [ "$stage1" != pending ] && \
     ! replicate_alive "$MID_STOP"; then
    if python3 "$HERE/full_pass.py" --ckpt-at "$MID_STOP" --root "$RUNS" \
         >/dev/null 2>&1; then
      # Card 1 must be clear of stage 1 first: one flock, one head at a time.
      if [ "$stage1" = done ]; then
        fire "$MID_STOP"
        stage2=fired
      fi
    fi
  fi

  if [ "$stage1" = done ] && [ "$stage2" = fired ] && \
     ! replicate_alive "$MID_STOP"; then
    log "both stages are drained — the queue stops"
    exit 0
  fi

  sleep "$PERIOD"
done
