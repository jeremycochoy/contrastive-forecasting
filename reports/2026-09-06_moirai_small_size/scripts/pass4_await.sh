#!/bin/bash
# #412 pass 4 — block until the pass ends, or until something needs a human.
#
# The agent session runs this as a background task. It returns on ONE of four
# conditions, and prints the reason as its last line:
#
#   COMPLETE    every leg holds a score, or its arm holds a collapse note
#   STALLED     a leg has no backbone, no trainer, and no lane
#   OVERTAKEN   a leg has no backbone, and a HIGHER stop of the same arm
#               already runs or landed. `phase1.sh` logs a failed leg and goes
#               on to the next stop, and the next stop resumes the arm's
#               FURTHEST checkpoint. So the skipped stop never gets a
#               checkpoint, and the card loses the comparison it rests on.
#   HEADBLOCKED a head is alive but its card is short of CF412_HEAD_VRAM_MIB,
#               and has been for CF412_HEAD_BLOCK_MIN minutes. Such a head
#               holds the card's head lock and aborts after 4 hours with no
#               score, so a reader sees a working head that produces nothing.
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
BLOCK_MIN="${CF412_HEAD_BLOCK_MIN:-30}"
# The blocked heads the session ALREADY knows about, as "<arm>@<N>k". A restart
# passes them back, so the await returns on a NEW block and not on an old one.
BLOCK_KNOWN="${CF412_BLOCK_KNOWN:-}"
EVERY="${CF412_AWAIT_EVERY:-120}"
LOG="$CF412_RESULTS/pass4_await.log"
mkdir -p "$CF412_RESULTS"

N_LEGS=0; for l in $LEGS; do N_LEGS=$(( N_LEGS + 1 )); done

step_of(){  # <arm> <stop>
  local f
  f="$(ls -t "$(cf412_leg_dir "$1" "$2")"/*_losses.csv 2>/dev/null | head -1)"
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

t0=$(date +%s); last_hb=$t0; last_move=$t0; prev_sum=-1; block_since=0
# The heads seen with a GPU allocation. A head runs its 97-config eval on the
# CPU after it drops the card, so it holds no GPU memory for hours while it
# works. Without this list that eval reads as a blocked head.
past_gate=""
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

  # A head that waits on memory is not a head that works.
  blocked=""; working=0
  for leg in $LEGS; do
    arm="${leg%%:*}"; stop="${leg##*:}"
    cf412_head_pid "$arm" "$stop" >/dev/null || continue
    key="$arm@$(( stop / 1000 ))k"
    case " $past_gate " in *" $key "*) working=$(( working + 1 )); continue ;; esac
    if b="$(cf412_head_blocked "$arm" "$stop")"; then
      blocked="$blocked $key gpu${b%% *}:${b##* }MiB"
    else
      past_gate="$past_gate $key"; working=$(( working + 1 ))
    fi
  done
  if [ -n "$blocked" ]; then
    [ "$block_since" -eq 0 ] && { block_since=$now
      say "a head waits on memory —$blocked"; }
    if [ $(( now - block_since )) -ge $(( BLOCK_MIN * 60 )) ]; then
      fresh=""
      for k in $blocked; do
        case "$k" in *@*) ;; *) continue ;; esac
        case " $BLOCK_KNOWN " in *" $k "*) ;; *) fresh="$fresh $k" ;; esac
      done
      if [ -n "$fresh" ]; then
        say "HEADBLOCKED for $BLOCK_MIN min —$fresh"
        say "  it holds the card head lock and aborts after 4 h with no score."
        echo HEADBLOCKED; exit 6
      fi
    fi
  else
    block_since=0
  fi

  # An overtaken leg. The card compares each arm against ITSELF at 40,000
  # steps, so a stop the lane skipped is not a delay, it is a lost comparison.
  for leg in $LEGS; do
    arm="${leg%%:*}"; stop="${leg##*:}"
    bb_of "$arm" "$stop" && continue
    busy_of "$arm" "$stop" && continue
    for hi in $LEGS; do
      [ "${hi%%:*}" = "$arm" ] || continue
      [ "${hi##*:}" -gt "$stop" ] || continue
      if bb_of "$arm" "${hi##*:}" || busy_of "$arm" "${hi##*:}"; then
        say "OVERTAKEN — $arm has no backbone at $stop, but ${hi##*:} runs or landed"
        say "  re-fire it: BB_GPU=<card> bash scripts/run_arm.sh $arm $stop"
        bash "$HERE/pass4_gate.sh" 2>&1 | tee -a "$LOG"; echo OVERTAKEN; exit 5
      fi
    done
  done

  # A stalled leg: not resolved, no backbone, no trainer, and no lane that
  # could still start it.
  for leg in $LEGS; do
    arm="${leg%%:*}"; stop="${leg##*:}"
    done_of "$arm" "$stop" && continue
    bb_of "$arm" "$stop" && continue
    busy_of "$arm" "$stop" && continue
    cf412_lane_for_arm "$arm" >/dev/null && continue
    say "STALLED — $arm at $stop has no backbone, no trainer, and no lane of its own"
    say "  re-fire it: BB_GPU=<card> bash scripts/run_arm.sh $arm $stop"
    bash "$HERE/pass4_gate.sh" 2>&1 | tee -a "$LOG"; echo STALLED; exit 3
  done

  if [ $(( now - last_move )) -ge $(( STALL_MIN * 60 )) ]; then
    if [ "$working" -eq 0 ]; then
      say "NOPROGRESS — no step for $STALL_MIN min and no head running —$line"
      echo NOPROGRESS; exit 4
    fi
    last_move=$now
  fi

  if [ $(( now - last_hb )) -ge 3600 ]; then
    last_hb=$now
    say "hourly — resolved $n of $N_LEGS —$line  lanes=$lanes heads=$heads working=$working"
  fi

  if [ $(( now - t0 )) -ge "$MAXWAIT" ]; then
    say "TIMEBOX — ${MAXWAIT}s passed, resolved $n of $N_LEGS —$line"
    bash "$HERE/pass4_gate.sh" 2>&1 | tee -a "$LOG"; echo TIMEBOX; exit 2
  fi
done
