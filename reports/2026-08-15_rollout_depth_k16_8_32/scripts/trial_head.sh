#!/bin/bash
# #401 — the whole pipeline, end to end, at a budget that finishes in minutes.
#
# Phase 1 spends 19 hours of backbone time before its first head runs. Every
# defect in the head half — a wrong `CF373_ROOT`, a score file written where
# collect.sh does not read, a phase-2 rule that refuses its own budget —
# would appear after that. This runs the same scripts on the same machine
# first, so the defect appears now.
#
# What runs, in order:
#
#   1. one backbone leg   k = 16, TRIAL_STEPS steps, through run_arm_k.sh
#                         and #373's run_leg_k.sh
#   2. phase 1            phase1.sh -> head_eval.sh -> #373's head_eval_bb.sh
#                         -> the head trainer -> eval_local.sh
#   3. collect.sh         over a REAL score file, both tables
#   4. phase 2            phase2.sh with the head budget matched to the stop
#
# Nothing about it is a simulation. It is the study's scripts, its runner,
# its head trainer and its eval, at a smaller budget:
#
#   CF401_TRIAL=<steps>     the stop, the phase-2 head budget, and the root
#                           and results suffix. See study.sh.
#   EVAL_CONFIG_FILTER      one GIFT-Eval config instead of 97. The protocol
#                           is unchanged, the config count is not, and the
#                           trial's score is therefore NOT comparable with a
#                           study number. It is a wiring check.
#   EVAL_EXPECT_CONFIGS=1   so eval_local.sh's merge count still holds.
#
# The trial writes to <root>-trial and to results/trial/, so no artefact of
# it can be collected as a study one.
#
# Usage:  BB_GPU=0 bash scripts/trial_head.sh
#         TRIAL_STEPS=400 TRIAL_K=16 bash scripts/trial_head.sh
set -uo pipefail

TRIAL_STEPS="${TRIAL_STEPS:-400}"
TRIAL_K="${TRIAL_K:-16}"
# `us_births/M/short` is the cheapest of the 97, at 0.0 s of the measured
# table. The regex is anchored, the same way shard_configs.py anchors its
# shards, so it matches one config and not a superstring of it.
TRIAL_CONFIG="${TRIAL_CONFIG:-^us_births/M/short$}"
BB_GPU="${BB_GPU:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CF401_TRIAL="$TRIAL_STEPS"
. "$HERE/study.sh"

export EVAL_CONFIG_FILTER="$TRIAL_CONFIG"
export EVAL_EXPECT_CONFIGS=1
export HEAD_BG=0          # inline, so a failure stops the trial where it is
mkdir -p "$CF401_RESULTS"

LOG="$CF401_RESULTS/trial.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 trial] $*" | tee -a "$LOG"; }

log "steps=$TRIAL_STEPS k=$TRIAL_K gpu=$BB_GPU config=$TRIAL_CONFIG"
log "root=$CF401_ROOT results=$CF401_RESULTS"
# The trial has one stop, and the card's phase-2 budget is the stop.
log "head budgets: phase 1 $CF401_HEAD_STEPS_P1, phase 2 $TRIAL_STEPS"

# Refuse to run against the study's own artefacts. The suffix comes from
# study.sh, and a trial that wrote into cf-401 would leave a 400-step
# checkpoint where a 40,000-step one belongs.
case "$CF401_ROOT" in
  *-trial) ;;
  *) echo "ABORT: CF401_ROOT=$CF401_ROOT is not a trial root" >&2; exit 2 ;;
esac
case "$CF401_RESULTS" in
  */trial) ;;
  *) echo "ABORT: CF401_RESULTS=$CF401_RESULTS is not a trial results dir" >&2
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

# 1. One backbone leg. `run_leg_k.sh` names its leg dir and its checkpoint
# from the target step count, so a TRIAL_STEPS below 1000 lands in leg_0k as
# `..._0k.pth` — which is exactly where cf401_bb_ckpt looks for this stop.
stage "backbone k=$TRIAL_K -> $TRIAL_STEPS steps" \
  env BB_GPU="$BB_GPU" bash "$HERE/run_arm_k.sh" "$TRIAL_K" "$TRIAL_STEPS"

BB="$(cf401_bb_ckpt "$TRIAL_K" "$TRIAL_STEPS")"
[ -n "$BB" ] && [ -f "$BB" ] || {
  log "ABORT: no checkpoint under $(cf401_leg_dir "$TRIAL_K" "$TRIAL_STEPS")"
  exit 3; }
log "backbone at $BB ($(du -h "$BB" | cut -f1))"

# 2. Phase 1 on that one arm: the head, then the one-config eval.
stage "phase 1" env DEPTHS="$TRIAL_K" BB_GPU="$BB_GPU" bash "$HERE/phase1.sh"

# 3. Both tables, over a real score file.
P1_TAG="$(cf401_tag "$TRIAL_K" "$TRIAL_STEPS" "$CF401_HEAD_STEPS_P1")"
[ -s "$CF401_RESULTS/score_${P1_TAG}.txt" ] || {
  log "ABORT: phase 1 wrote no score at $CF401_RESULTS/score_${P1_TAG}.txt"
  exit 4; }
stage "collect" bash "$HERE/collect.sh"

# 4. Phase 2 on the same arm, head budget matched to the stop. ARMS skips
# the picker, which needs all three depths and has its own unit tests.
stage "phase 2" env ARMS="$TRIAL_K" BB_GPU="$BB_GPU" bash "$HERE/phase2.sh"

P2_TAG="$(cf401_tag "$TRIAL_K" "$TRIAL_STEPS" "$TRIAL_STEPS")"
[ "$P1_TAG" != "$P2_TAG" ] || {
  log "ABORT: both phases wrote the tag $P1_TAG"; exit 5; }
[ -s "$CF401_RESULTS/score_${P2_TAG}.txt" ] || {
  log "ABORT: phase 2 wrote no score at $CF401_RESULTS/score_${P2_TAG}.txt"
  exit 4; }

log "--- $CF401_RESULTS/scores.csv ---"
cat "$CF401_RESULTS/scores.csv" | tee -a "$LOG"
if [ -s "$CF401_RESULTS/splits.csv" ]; then
  log "--- $CF401_RESULTS/splits.csv ($(( $(wc -l <"$CF401_RESULTS/splits.csv") - 1 )) rows) ---"
  head -8 "$CF401_RESULTS/splits.csv" | tee -a "$LOG"
else
  log "WARNING: no splits.csv — the per-domain deliverable has no input"
fi
log "TRIAL PASSED"
