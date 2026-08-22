#!/bin/bash
# #409 — ONE lane: the arms this card gives one card, in order.
#
# One arm is one backbone to 40,000 steps, then one 30,000-step student head,
# then that head's 97 GIFT-Eval configs. The arms are independent — no arm
# resumes another — so a lane can take any subset. `launch.sh` deals the eight
# arms over elisa's two cards and starts one lane on each.
#
# ---- A crash is re-fired, a refusal is not -----------------------------------
#
# `run_leg_k.sh` resumes the cell's furthest checkpoint with its optimizer
# state, so a re-fired leg costs only the steps since the last save. This lane
# re-fires a leg CF409_LEG_TRIES times.
#
# It never re-fires a refusal. `CF409_NO_RETRY` in study.sh lists the codes: an
# unknown arm (2), an objective that did not reach the trainer (3), an arm the
# AUC gate stopped (4), a session hold (9) and a cell another machine claims
# (10). Each of those repeats, and each re-fire costs a card.
#
# ---- Where the head runs -----------------------------------------------------
#
# Each arm's head starts as soon as its checkpoint lands, while the next arm
# trains. Head training and backbone training both want the card, so
# `head_eval_bb.sh` waits for free VRAM and the GIFT-Eval that follows runs on
# the CPU. HEAD_BG=0 runs each head inline instead, which is slower to drain
# and easier to read.
#
# CF409_HEADS=0 trains the backbones and nothing else.
#
# Usage:  bash phase1.sh                          # every arm of the card
#         ARMS="dec0_s20 flr05_s20" bash phase1.sh
#         BB_GPU=1 ARMS=dec0_s20 bash phase1.sh
#         CF409_DRY_RUN=1 bash phase1.sh          # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARMS="${ARMS:-$CF409_ARMS}"
HEAD_BG="${HEAD_BG:-1}"
HEADS="${CF409_HEADS:-1}"
BB_GPU="${BB_GPU:-0}"
HEAD_GPU="${HEAD_GPU:-$BB_GPU}"
# The two legs of one arm, as paths. A test replaces them with stubs, so this
# lane's retry rule can be exercised without a card.
RUN_ARM="${CF409_RUN_ARM:-$HERE/run_arm.sh}"
HEAD_EVAL="${CF409_HEAD_EVAL:-$HERE/head_eval.sh}"
mkdir -p "$CF409_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409 lane $BB_GPU] $*" \
  | tee -a "$CF409_RESULTS/phase1.log"; }

# One arm's backbone, re-fired while the exit code says "crash". Prints the
# final code.
run_leg(){  # <arm> <stop steps>
  local arm="$1" stop="$2" try=1 rc
  while :; do
    BB_GPU="$BB_GPU" bash "$RUN_ARM" "$arm" "$stop"
    rc=$?
    [ "$rc" -eq 0 ] && return 0
    cf409_retryable "$rc" || return "$rc"
    [ "$try" -ge "$CF409_LEG_TRIES" ] && return "$rc"
    try=$(( try + 1 ))
    log "arm $arm rc=$rc — re-firing (try $try of $CF409_LEG_TRIES)." \
        "The leg resumes the cell's furthest checkpoint."
    sleep "${CF409_LEG_RETRY_WAIT:-60}"
  done
}

heads=(); head_names=(); inline_failed=0; legs_failed=0
for arm in $ARMS; do
  cf409_require_arm "$arm" || exit $?
  for stop in $CF409_STOPS; do
    if [ -n "${CF409_DRY_RUN:-}" ]; then
      echo "arm $arm steps=$stop gpu=$BB_GPU tries=$CF409_LEG_TRIES" \
           "decay=$(cf409_decay_args "$arm")"
      [ "$HEADS" = "1" ] && \
        echo "head $arm stop=$stop steps=$CF409_HEAD_STEPS enc=$CF409_ENC"
      continue
    fi

    log "arm $arm -> $stop"
    run_leg "$arm" "$stop"
    rc=$?
    if [ $rc -ne 0 ]; then
      # The count reaches the exit status. A lane that reported success here
      # would leave a dead arm to surface hours later, as a table with seven
      # rows and no reason for the eighth.
      legs_failed=$(( legs_failed + 1 ))
      if [ "$rc" -eq "$CF409_RC_COLLAPSED" ]; then
        log "arm $arm STOPPED by the AUC gate — no head, no score." \
            "See $(cf409_collapse_file "$arm")"
      else
        log "arm $arm stop=$stop rc=$rc — no head for this arm"
      fi
      continue
    fi

    if [ "$HEADS" != "1" ]; then
      log "head $arm SKIPPED (CF409_HEADS=0)"
    elif [ "$HEAD_BG" = "1" ]; then
      log "head $arm stop=$stop (background)"
      # `nohup`, not `nohup setsid`. setsid forks when the caller already leads
      # its process group, and then `$!` is the PID of a process that exits at
      # once — so the `wait` below would return before the head had started.
      BB_GPU="$HEAD_GPU" nohup bash "$HEAD_EVAL" "$arm" "$stop" \
        >>"$CF409_RESULTS/head_${arm}_bb$(cf409_steps_label "$stop").out" 2>&1 &
      heads+=($!); head_names+=("$arm stop=$stop")
    else
      log "head $arm stop=$stop (inline)"
      BB_GPU="$HEAD_GPU" bash "$HEAD_EVAL" "$arm" "$stop"
      rc=$?
      log "head $arm stop=$stop rc=$rc"
      [ $rc -eq 0 ] || inline_failed=$(( inline_failed + 1 ))
    fi
  done
done

[ -n "${CF409_DRY_RUN:-}" ] && exit 0

# A head still running when the last backbone finishes is the normal case, and
# its GIFT-Eval is hours of CPU. Waiting here is what makes `collect.sh`
# afterwards see every score.
failed="$inline_failed"
if [ "${#heads[@]}" -gt 0 ]; then
  log "waiting for ${#heads[@]} head+eval job(s)"
  for i in "${!heads[@]}"; do
    wait "${heads[$i]}"; rc=$?
    if [ $rc -eq 0 ]; then
      log "head ${head_names[$i]} rc=0"
    else
      failed=$(( failed + 1 ))
      log "head ${head_names[$i]} rc=$rc — see head_*.out in $CF409_RESULTS"
    fi
  done
fi

# A backbone-only lane scores nothing, so there is nothing to collect. Running
# collect.sh there would write an empty scores.csv over the one the lane that
# DOES score keeps.
if [ "$HEADS" = "1" ]; then
  bash "$HERE/collect.sh" >>"$CF409_RESULTS/phase1.log" 2>&1
fi
log "lane drained — $legs_failed leg(s) and $failed head(s) failed"
failed=$(( failed + legs_failed ))
[ "$failed" -eq 0 ] || exit 1
