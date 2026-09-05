#!/bin/bash
# #409 — ONE lane: the arms this card gives one card, in order. One arm is one
# EMA schedule.
#
# One arm is one backbone to 40,000 steps, then one 30,000-step student head,
# then that head's 97 GIFT-Eval configs. The arms are independent. No arm
# resumes another, so a lane can take any subset. `launch.sh` deals the arms
# over elisa's two cards and starts one lane on each.
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
# ---- A Hub outage costs no try -----------------------------------------------
#
# The data streams from the Hub. On 08-23 at 18:48 elisa lost DNS, every leg
# died in about 3 seconds, and this lane spent an arm's whole ladder in two
# minutes. It then declared a FRESH start, moved to the next arm and did the
# same. Three arms went in seven minutes, and the card sat idle for 27 hours.
#
# So the Hub is not the arm. A leg that exits `CF409_RC_NETWORK` is re-fired
# after a growing wait, and the try is NOT counted. The lane also reads the
# Hub before it starts any arm, so it never advances into a dead network. It
# gives up only at `CF409_NET_DEADLINE`, which is hours.
#
# ---- Where the head runs -----------------------------------------------------
#
# Each arm's head starts as soon as its checkpoint lands, while the next arm
# trains. Head training and backbone training both want the card, so
# `head_eval_bb.sh` waits for free VRAM and the GIFT-Eval that follows runs on
# the CPU.
#
# A lane holds several arms, so several heads would pile onto one card. This
# lane holds ONE: it queues each head and a single worker runs them in turn. The
# queue does not hold the card idle — the next backbone starts the moment the
# one before it ends, whatever the worker is doing.
#
# HEAD_BG=0 runs each head inline instead, which is slower to drain and easier
# to read. CF409_HEADS=0 trains the backbones and nothing else.
#
# Usage:  bash phase1.sh                          # every arm of the card
#         ARMS="dec_s20 dec_m090_fix" bash phase1.sh
#         BB_GPU=1 ARMS=dec_m090_fix bash phase1.sh
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

# One arm is refused before any head is queued, so a bad name never leaves a
# worker holding heads this lane will not wait for.
for arm in $ARMS; do cf409_require_arm "$arm" || exit $?; done

# One arm's backbone, re-fired while the exit code says "crash". Prints the
# final code.
run_leg(){  # <arm> <stop steps>
  local arm="$1" stop="$2" try=1 rc net_try=0 net_spent=0 delay
  while :; do
    # `8>&-` closes the write end of the head queue for this child. A child
    # that held it open would keep the worker waiting for a line that never
    # comes.
    BB_GPU="$BB_GPU" bash "$RUN_ARM" "$arm" "$stop" 8>&-
    rc=$?
    [ "$rc" -eq 0 ] && return 0
    if [ "$rc" -eq "$CF409_RC_NETWORK" ]; then
      # The arm is fine, so this costs no try: a 3-second death to a dead
      # network must never spend a try that a real crash needs.
      #
      # It does cost TIME, and the two are not the same. A Hub that answers
      # the probe while every leg still dies on a shard would re-fire for
      # ever and hold the card at zero steps. So one leg gets at most
      # CF409_NET_DEADLINE seconds of outage, over a growing delay.
      net_try=$(( net_try + 1 ))
      delay="$(hub_backoff_delay "$net_try")"
      net_spent=$(( net_spent + delay ))
      if [ "$net_spent" -gt "$CF409_NET_DEADLINE" ]; then
        log "arm $arm — the Hub has killed this leg for ${net_spent}s, past" \
            "the ${CF409_NET_DEADLINE}s deadline. The arm keeps its" \
            "checkpoints, so a later lane resumes it."
        return "$rc"
      fi
      log "arm $arm — the Hub was unreachable. This costs no try" \
          "(still try $try of $CF409_LEG_TRIES). Waiting ${delay}s."
      sleep "$delay"
      hub_wait_up "$(( CF409_NET_DEADLINE - net_spent ))" \
        | tee -a "$CF409_RESULTS/phase1.log" || return "$rc"
      continue
    fi
    cf409_retryable "$rc" || return "$rc"
    [ "$try" -ge "$CF409_LEG_TRIES" ] && return "$rc"
    try=$(( try + 1 ))
    log "arm $arm rc=$rc — re-firing (try $try of $CF409_LEG_TRIES)." \
        "The leg resumes the cell's furthest checkpoint."
    sleep "${CF409_LEG_RETRY_WAIT:-60}"
  done
}

# The lane's head worker: one head at a time, in the order the lane queues
# them. It reads `<arm> <stop>` lines on stdin and ends at end of file, which
# the lane sends by closing the write end of the queue.
head_worker(){
  local arm stop rc failed=0
  # The old code put each head under `nohup`. This keeps that: a hangup on the
  # lane must not stop a head that holds hours of GIFT-Eval.
  trap '' HUP
  while read -r arm stop; do
    log "head $arm stop=$stop start"
    BB_GPU="$HEAD_GPU" bash "$HEAD_EVAL" "$arm" "$stop" \
      >>"$CF409_RESULTS/head_${arm}_bb$(cf409_steps_label "$stop").out" \
      2>&1 </dev/null
    rc=$?
    log "head $arm stop=$stop rc=$rc"
    [ "$rc" -eq 0 ] || failed=$(( failed + 1 ))
  done
  return "$failed"
}

# A named pipe, not a file: the worker blocks on `read` between heads and needs
# no poll. The worker opens the read end first, so the lane's own open returns.
#
# The name carries the lane's PID, not the card index alone. This cell holds
# the card at about 26 percent, so elisa runs SEVERAL lanes on one card, and
# two lanes with one queue name would each delete the other's pipe.
worker=""; queue=""
if [ "$HEADS" = "1" ] && [ "$HEAD_BG" = "1" ] && [ -z "${CF409_DRY_RUN:-}" ]; then
  queue="$CF409_RESULTS/head_queue_gpu${BB_GPU}_$$"
  rm -f "$queue"
  if mkfifo "$queue" 2>/dev/null; then
    head_worker <"$queue" &
    worker=$!
    exec 8>"$queue"
    log "head queue on — one head at a time, worker pid $worker"
  else
    log "no queue at $queue — the heads run inline"
    HEAD_BG=0
  fi
fi

inline_failed=0; legs_failed=0
for arm in $ARMS; do
  for stop in $CF409_STOPS; do
    if [ -n "${CF409_DRY_RUN:-}" ]; then
      echo "arm $arm steps=$stop gpu=$BB_GPU tries=$CF409_LEG_TRIES" \
           "ema='$(cf409_ema_label "$arm")'" \
           "seed=$(cf409_seed "$arm") decay=$(cf409_decay_args "$arm")"
      [ "$HEADS" = "1" ] && \
        echo "head $arm stop=$stop steps=$CF409_HEAD_STEPS enc=$CF409_ENC"
      continue
    fi

    # A lane that starts an arm into a dead network burns that arm's ladder
    # in two minutes and moves on. Read the Hub first.
    if ! hub_wait_up "$CF409_NET_DEADLINE" \
         | tee -a "$CF409_RESULTS/phase1.log"; then
      log "the Hub is still down — starting no arm. $arm and the arms after" \
          "it keep their checkpoints, so a later lane resumes them."
      legs_failed=$(( legs_failed + 1 ))
      break 2
    fi

    log "arm $arm -> $stop"
    run_leg "$arm" "$stop"
    rc=$?
    if [ $rc -ne 0 ]; then
      # The count reaches the exit status. A lane that reported success here
      # would leave a dead arm to surface hours later, as a table with one row
      # missing and no reason for it.
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
    elif [ -n "$worker" ]; then
      log "head $arm stop=$stop (queued)"
      echo "$arm $stop" >&8
    else
      log "head $arm stop=$stop (inline)"
      BB_GPU="$HEAD_GPU" bash "$HEAD_EVAL" "$arm" "$stop" 8>&-
      rc=$?
      log "head $arm stop=$stop rc=$rc"
      [ $rc -eq 0 ] || inline_failed=$(( inline_failed + 1 ))
    fi
  done
done

[ -n "${CF409_DRY_RUN:-}" ] && exit 0

# A head still running when the last backbone finishes is the normal case, and
# its GIFT-Eval is hours of CPU. Waiting here is what makes `collect.sh`
# afterwards see every score. The worker ends at end of file, which is what
# closing fd 8 sends.
failed="$inline_failed"
if [ -n "$worker" ]; then
  exec 8>&-
  log "waiting for the head queue to drain"
  wait "$worker"; rc=$?
  rm -f "$queue"
  [ "$rc" -eq 0 ] || log "$rc head(s) failed — see head_*.out in $CF409_RESULTS"
  failed=$(( failed + rc ))
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
