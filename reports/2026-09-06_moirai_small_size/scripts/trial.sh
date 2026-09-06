#!/bin/bash
# #412 — the whole pipeline, end to end, at a budget that ends in minutes.
#
# `smoke.sh` proves the BACKBONE half: the width, the depth and the momentum
# reach the trainer. It proves nothing about the head half, and the head half
# is where this card's width bites hardest.
#
# The head trainer and the GIFT-Eval REBUILD the backbone before they load its
# weights, and both take the shape from the command line. Both held
# `--d-model 64` as a literal. At 384 they cannot load this card's checkpoint,
# and phase 1 would spend a backbone leg before it found out. This runs the
# same scripts on the same machine first.
#
# What runs, in order:
#
#   1. one backbone leg   TRIAL_STEPS steps, through run_arm.sh and #373's
#                         run_leg_k.sh, at d_model 384
#   2. phase 1            phase1.sh -> head_eval.sh -> #373's head_eval_bb.sh
#                         -> the head trainer -> eval_local.sh
#   3. collect.sh         over a REAL score file
#
# Nothing about it is a simulation. It is the study's scripts, its runner, its
# head trainer and its evaluation, at a smaller budget:
#
#   CF412_TRIAL=<steps>     the stop, the head budget, and the root and
#                           results suffix. See study.sh.
#   EVAL_CONFIG_FILTER      one GIFT-Eval config instead of 97. The protocol
#                           is unchanged, the config count is not, so the
#                           trial's score is NOT comparable with a study
#                           number. It is a wiring check.
#   EVAL_EXPECT_CONFIGS=1   so eval_local.sh's merge count still holds.
#
# The trial writes to <root>-trial and to results/trial/, so no artefact of it
# can be collected as a study one.
#
# Usage:  BB_GPU=0 bash scripts/trial.sh
#         TRIAL_STEPS=100 TRIAL_ARM=k32_r100_09 bash scripts/trial.sh
set -uo pipefail

TRIAL_STEPS="${TRIAL_STEPS:-40}"
TRIAL_ARM="${TRIAL_ARM:-k3_r100_09}"
# `us_births/M/short` is the cheapest of the 97 configs. The regex is
# anchored, so it matches one config and not a longer name that holds it.
TRIAL_CONFIG="${TRIAL_CONFIG:-^us_births/M/short$}"
BB_GPU="${BB_GPU:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CF412_TRIAL="$TRIAL_STEPS"
. "$HERE/study.sh"

export EVAL_CONFIG_FILTER="$TRIAL_CONFIG"
export EVAL_EXPECT_CONFIGS=1
export LOG_EVERY="${LOG_EVERY:-10}"
mkdir -p "$CF412_RESULTS"

LOG="$CF412_RESULTS/trial.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 trial] $*" | tee -a "$LOG"; }

log "arm=$TRIAL_ARM steps=$TRIAL_STEPS gpu=$BB_GPU config=$TRIAL_CONFIG"
log "shape=$(cf412_arch_args)"
log "root=$CF412_ROOT results=$CF412_RESULTS"

# Refuse to run against the study's own artefacts. The suffix comes from
# study.sh, and a trial that wrote into cf-412 would leave a 40-step
# checkpoint where a 40,000-step one belongs.
case "$CF412_ROOT" in
  *-trial) ;;
  *) echo "ABORT: CF412_ROOT=$CF412_ROOT is not a trial root" >&2; exit 2 ;;
esac
case "$CF412_RESULTS" in
  */trial) ;;
  *) echo "ABORT: CF412_RESULTS=$CF412_RESULTS is not a trial results dir" >&2
     exit 2 ;;
esac

stage(){  # <name> <command...>
  local name="$1"; shift
  log "START $name"
  "$@"
  local rc=$?
  log "$name rc=$rc"
  [ $rc -eq 0 ] || { log "TRIAL FAILED at $name"; exit $rc; }
}

# Phase 1 is the backbone leg and its head and its evaluation, in one call.
# `run_leg_k.sh` names its leg directory and its checkpoint from the target
# step count, so a TRIAL_STEPS below 1000 lands in leg_0k as `..._0k.pth`,
# which is where cf412_bb_ckpt looks for this stop.
stage "phase 1 on $TRIAL_ARM" \
  env ARMS="$TRIAL_ARM" BB_GPU="$BB_GPU" bash "$HERE/phase1.sh"

BB="$(cf412_bb_ckpt "$TRIAL_ARM" "$TRIAL_STEPS")"
[ -n "$BB" ] && [ -f "$BB" ] || {
  log "ABORT: no checkpoint under $(cf412_leg_dir "$TRIAL_ARM" "$TRIAL_STEPS")"
  exit 3; }
log "backbone at $BB ($(du -h "$BB" | cut -f1))"

TAG="$(cf412_tag "$TRIAL_ARM" "$TRIAL_STEPS" "$CF412_HEAD_STEPS")"
SCORE="$(cf412_score_file "$TRIAL_ARM" "$TRIAL_STEPS")"
[ -s "$SCORE" ] || { log "ABORT: phase 1 wrote no score at $SCORE"; exit 4; }
log "score $TAG = $(cat "$SCORE")"

stage "collect" bash "$HERE/collect.sh"
log "TRIAL PASSED"
