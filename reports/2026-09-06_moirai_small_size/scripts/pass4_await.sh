#!/bin/bash
# #412 pass 4 — block until the pass ends, or until something needs a human.
#
# The agent session runs this as a background task. It returns on ONE of four
# conditions, and prints the reason as its last line:
#
#   COMPLETE    every leg holds a score, or its arm holds a collapse note
#   STALLED     a leg has no backbone, no trainer, and no lane
#   NOPROGRESS  no leg advanced a step for CF412_STALL_MIN minutes
#   TIMEBOX     CF412_MAXWAIT seconds passed, so the session re-arms it
#
# It does not poll inside the agent turn. It sleeps and prints one status line
# every hour into `results/pass4_await.log`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
. "$HERE/pass4_lib.sh"

LEGS="${CF412_PASS4_LEGS:-k3_r100_09_lr56_dec:40000 k3_r100_09_lr56_dec:100000 k3_r100_09_lr56_dec:200000 k3_r100_09_lr56_dec10k:40000 k3_r100_09_lr56_dec10k:100000}"
MAXWAIT="${CF412_MAXWAIT:-14400}"
STALL_MIN="${CF412_STALL_MIN:-60}"
EVERY="${CF412_AWAIT_EVERY:-120}"
LOG="$CF412_RESULTS/pass4_await.log"
mkdir -p "$CF412_RESULTS"

N_LEGS=0; for l in $LEGS; do N_LEGS=$(( N_LEGS + 1 )); done

step_of(){  # <arm> <stop>
  local f
  f="$(ls "$(cf412_leg_dir "$1" "$2")"/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$f" ] || { echo 0; return 0; }
  tail -1 "$f" | cut -d, -f1 | grep -E '^[0-9]+$' || echo 0
}
# A leg is resolved when it holds a score, or when its arm lost the
# contrastive task. A lost arm does not climb, so its higher stops never run.
done_of(){  # <arm> <stop>
  [ -s "$(cf412_score_file "$1" "$2")" ] && return 0
  [ -f "$(cf412_collapse_file "$1")" ] && return 0
  return 1
}
bb_of(){ local b; b="$(cf412_bb_ckpt "$1" "$2")"; [ -n "$b" ] && [ -f "$b" ]; }
busy_of(){  # <arm> <stop>
  cf412_trainer_pid "$1" "$2" >/dev/null || cf412_head_pid "$1" "$2" >/dev/null
}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

t0=$(date +%s); last_hb=$t0; last_move=$t0; prev_sum=-1
say "await armed — $N_LEGS legs, maxwait ${MAXWAIT}s, stall ${STALL_MIN}min"
while :; do
  sleep "$EVERY"
  now=$(date +%s)

  n=0; sum=0; line=""
  for leg in $LEGS; do
    arm="${leg%%:*}"; stop="${leg##*:}"
    done_of "$arm" "$stop" && n=$(( n + 1 ))
    s="$(step_of "$arm" "$stop")"; sum=$(( sum + s ))
    line="$line ${arm#k3_r100_09_lr56_}:$(( stop / 1000 ))k=$s"
  done

  [ "$n" -ge "$N_LEGS" ] && { say "every leg is resolved"
    bash "$HERE/pass4_gate.sh" 2>&1 | tee -a "$LOG"; echo COMPLETE; exit 0; }

  [ "$sum" -ne "$prev_sum" ] && { prev_sum=$sum; last_move=$now; }

  lanes=$(cf412_lanes_running)
  heads=$(cf412_heads_running)

  # A stalled leg: not resolved, no backbone, no trainer, and no lane that
  # could still start it.
  for leg in $LEGS; do
    arm="${leg%%:*}"; stop="${leg##*:}"
    done_of "$arm" "$stop" && continue
    bb_of "$arm" "$stop" && continue
    busy_of "$arm" "$stop" && continue
    [ "$lanes" -gt 0 ] && continue
    say "STALLED — $arm at $stop has no backbone, no trainer, no lane"
    bash "$HERE/pass4_gate.sh" 2>&1 | tee -a "$LOG"; echo STALLED; exit 3
  done

  if [ $(( now - last_move )) -ge $(( STALL_MIN * 60 )) ]; then
    if [ "$heads" -eq 0 ]; then
      say "NOPROGRESS — no step for $STALL_MIN min and no head running —$line"
      echo NOPROGRESS; exit 4
    fi
    last_move=$now
  fi

  if [ $(( now - last_hb )) -ge 3600 ]; then
    last_hb=$now
    say "hourly — resolved $n of $N_LEGS —$line  lanes=$lanes heads=$heads"
  fi

  if [ $(( now - t0 )) -ge "$MAXWAIT" ]; then
    say "TIMEBOX — ${MAXWAIT}s passed, resolved $n of $N_LEGS —$line"
    bash "$HERE/pass4_gate.sh" 2>&1 | tee -a "$LOG"; echo TIMEBOX; exit 2
  fi
done
