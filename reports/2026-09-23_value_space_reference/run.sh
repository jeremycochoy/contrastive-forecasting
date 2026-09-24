#!/bin/bash
# #415 — the whole leg: train the value-space reference to each stop, and
# score each stop under B4 and A2.
#
# Usage:  BB_GPU=0 bash reports/2026-09-23_value_space_reference/run.sh
#         CF415_DRY_RUN=1 bash .../run.sh        # print every command
#         bash .../run.sh 40000 100000           # a subset of the stops
#
# The card trains ONE model and scores it eight times. Each stop resumes the
# previous stop's checkpoint with its optimizer state, so the eight points sit
# on one trajectory — the shape #414 measures. 665,000 steps is one pass over
# the data.
#
# Two lanes, so no score holds up the training:
#
#   train   trains the stops in order, in the background.
#   score   scores the stops in the same order, each one as soon as its
#           checkpoint is on disk: one head on the GPU, then the 97-config
#           GIFT-Eval on the CPU, under B4 and then under A2.
#
# A2 re-runs the whole cell once per forecast patch. On one core it costs 3.9
# times B4 per series at horizons 480 and 720 (`scripts/a2_cost.py`). A
# ladder that scored each stop before it trained the next would leave the
# card idle for most of the run.
#
# This script is the one lane that scores its stops. Do not start a second
# scorer on a stop it covers: two writers in one shard directory abort the
# merge.
#
# Both lanes are idempotent: a stop whose checkpoint is on disk trains
# nothing, and a strategy whose score file exists scores nothing. So a
# re-fire after a crash costs only what was lost.
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/scripts/paths.sh"
. "$CF415_PARENT/scripts/leg_paths.sh"

# The eight stops live in `paths.sh`, because `head_eval_value.sh` validates
# against them. A second list here would let a stop train and then fail to
# score.
STOPS="$CF415_STOPS"
[ "$#" -gt 0 ] && STOPS="$*"

BB_GPU="${BB_GPU:-0}"
# The two halves. Variables so a test can hand the ladder stubs and prove the
# lanes overlap. The card never sets them.
LEG_RUNNER="${CF415_LEG_RUNNER:-$CF415_SCRIPTS/run_leg_value.sh}"
SCORER="${CF415_SCORER:-$CF415_SCRIPTS/head_eval_value.sh}"
# Seconds between two looks for the next checkpoint.
POLL="${CF415_POLL:-60}"

# A dry run prints and writes nothing, so the guards can run the whole ladder
# without leaving a log behind in a shared checkout.
if [ -n "${CF415_DRY_RUN:-}" ]; then
  for stop in $STOPS; do
    BB_GPU="$BB_GPU" bash "$LEG_RUNNER" "$stop" || exit $?
    BB_GPU="$BB_GPU" bash "$SCORER" "$stop" || exit $?
  done
  exit 0
fi

mkdir -p "$CF415_RESULTS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#415] $*" \
  | tee -a "$CF415_RESULTS/run.log"; }

# The trainer writes the weights and then the optimizer state, and neither
# write is atomic. So the optimizer file on disk says the weights are whole.
stop_ready(){  # <stop steps>
  local ckpt
  ckpt="$(cf415_bb_ckpt "$1")"
  [ -n "$ckpt" ] && [ -f "${ckpt%.pth}_optimizer.pth" ]
}

train_lane(){
  local stop rc
  for stop in $STOPS; do
    log "stop $stop: train"
    BB_GPU="$BB_GPU" bash "$LEG_RUNNER" "$stop" || {
      rc=$?; log "stop $stop: train rc=$rc — the train lane stops"; return $rc; }
  done
  log "train lane done"
}

score_lane(){  # <pid of the train lane>
  local stop rc=0
  for stop in $STOPS; do
    until stop_ready "$stop"; do
      if ! kill -0 "$1" 2>/dev/null; then
        # One last look: the lane can end just after it wrote this stop.
        stop_ready "$stop" && break
        log "stop $stop: no checkpoint with its optimizer file — the score lane stops"
        return 1
      fi
      sleep "$POLL"
    done
    log "stop $stop: head + 97-config GIFT-Eval, B4 then A2"
    if BB_GPU="$BB_GPU" bash "$SCORER" "$stop"; then
      log "stop $stop: GM-Relative MASE" \
          "B4 $(cat "$(cf415_score_file "$stop" B4)" 2>/dev/null)" \
          "A2 $(cat "$(cf415_score_file "$stop" A2)" 2>/dev/null)"
    else
      rc=$?; log "stop $stop: score rc=$rc — the next stop still scores"
    fi
  done
  return $rc
}

train_lane & train_pid=$!
score_lane "$train_pid"; score_rc=$?
wait "$train_pid"; train_rc=$?
if [ "$train_rc" -ne 0 ] || [ "$score_rc" -ne 0 ]; then
  log "ENDED train rc=$train_rc score rc=$score_rc"
  [ "$train_rc" -ne 0 ] && exit "$train_rc"
  exit "$score_rc"
fi
log "DONE — scores in $CF415_RESULTS/score_*.txt"
