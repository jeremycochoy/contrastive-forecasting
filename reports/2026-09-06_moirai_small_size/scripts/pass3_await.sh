#!/bin/bash
# #412 pass 3 — block until the pass ends, or until something needs a human.
#
# The agent session runs this as a background task. It returns on ONE of four
# conditions, and prints the reason as its last line:
#
#   COMPLETE    every pass-3 arm holds a score or a collapse note
#   STALLED     an arm has no lane, no trainer, no 40,000-step backbone
#   NOPROGRESS  no arm advanced a step for CF412_STALL_MIN minutes
#   TIMEBOX     CF412_MAXWAIT seconds passed, so the session re-arms it
#
# It does not poll inside the agent turn. It sleeps and prints one status
# line every hour into `results/pass3_await.log`.
set -uo pipefail
D=/tmp/contrastive-forecasting-412/reports/2026-09-06_moirai_small_size
R="$D/results"
CK=/home/jupyter/checkpoints_backup/cf-412
ARMS="${CF412_PASS3_ARMS:-k32_r100_09_sum k3_r100_09_lr45 k3_r100_09_lr70}"
MAXWAIT="${CF412_MAXWAIT:-21600}"
STALL_MIN="${CF412_STALL_MIN:-45}"
EVERY=120
LOG="$R/pass3_await.log"

step_of(){ local f
  f="$(ls "$CK/$1/arm6_v2_combab_alignT/leg_40k/"*_losses.csv 2>/dev/null | head -1)"
  [ -n "$f" ] && tail -1 "$f" | cut -d, -f1 | grep -E '^[0-9]+$' || echo 0; }
done_of(){ [ -s "$R/score_${1}_bb40k_h30k_student.txt" ] || [ -f "$R/collapsed_$1.txt" ]; }
bb_of(){ ls "$CK/$1/arm6_v2_combab_alignT/leg_40k/"*_40k.pth >/dev/null 2>&1; }
busy(){ ps -eo args --no-headers 2>/dev/null | grep -c "[c]f412_$1\|[/]$1/arm6" ; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

t0=$(date +%s); last_hb=$t0; last_move=$t0; prev_sum=-1
say "await armed — arms: $ARMS  maxwait ${MAXWAIT}s"
while :; do
  sleep "$EVERY"
  now=$(date +%s)

  n=0; sum=0; line=""
  for a in $ARMS; do
    done_of "$a" && n=$(( n + 1 ))
    s="$(step_of "$a")"; sum=$(( sum + s )); line="$line $a=$s"
  done
  [ "$n" -ge 3 ] && { say "every arm is done"; bash "$D/scripts/gate_pass3.sh"; echo COMPLETE; exit 0; }

  [ "$sum" -ne "$prev_sum" ] && { prev_sum=$sum; last_move=$now; }

  # A stalled arm: not done, no 40,000-step backbone, and nothing running.
  for a in $ARMS; do
    done_of "$a" && continue
    bb_of "$a" && continue
    [ "$(busy "$a")" -gt 0 ] && continue
    lanes=$(ps -eo args --no-headers | grep -c "[p]ass2_lane.sh")
    heads=$(ps -eo args --no-headers | grep -c "[h]ead_eval")
    [ "$lanes" -gt 0 ] || [ "$heads" -gt 0 ] && continue
    say "STALLED — $a has no backbone, no lane, no trainer"
    bash "$D/scripts/gate_pass3.sh"; echo STALLED; exit 3
  done

  if [ $(( now - last_move )) -ge $(( STALL_MIN * 60 )) ] && [ "$n" -lt 3 ]; then
    heads=$(ps -eo args --no-headers | grep -c "[h]ead_eval")
    if [ "$heads" -eq 0 ]; then
      say "NOPROGRESS — no step for $STALL_MIN min and no head running —$line"
      echo NOPROGRESS; exit 4
    fi
    last_move=$now
  fi

  if [ $(( now - last_hb )) -ge 3600 ]; then
    last_hb=$now
    say "hourly — scored/collapsed $n of 3 —$line  heads=$(ps -eo args --no-headers | grep -c '[h]ead_eval')"
  fi

  if [ $(( now - t0 )) -ge "$MAXWAIT" ]; then
    say "TIMEBOX — ${MAXWAIT}s passed, scored/collapsed $n of 3 —$line"
    bash "$D/scripts/gate_pass3.sh"; echo TIMEBOX; exit 2
  fi
done
