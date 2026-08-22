#!/bin/bash
# #409 — one student head on one arm's backbone, and its 97-config GIFT-Eval.
#
# The card's first deliverable is a GIFT-Eval score for every arm. The backbone
# leg (`run_arm.sh`) stops at the checkpoint. This is the rest of the path.
#
# #373's `head_eval_bb.sh` already takes an EXPLICIT backbone path and a head
# budget, for exactly this reason: a backbone that its own cell table does not
# name. So this wrapper resolves the checkpoint, builds the tag and hands both
# over. The protocol below it is #373's, unchanged — quantile head, 2-layer
# transformer, forecast-len 16, batch 256, lr 1e-3, head seed 20260722, then
# the 97 GIFT-Eval configs under strategy B4 on the CPU.
#
# The budget is 30,000 head steps, which is what #401 and #404 ran on a bb40k
# backbone. The card's published reference, 1.0862, came from a 15,000-step
# head, so the two control arms are what this card compares its treated arms
# against. #401 measured that a longer head moves the score.
#
# Idempotent. A scored tag is a no-op, and a trained head skips to its eval.
#
# Usage:  head_eval.sh <arm> <stop steps> [head steps]
#         BB_GPU=0 bash head_eval.sh dec0_s20 40000
#         CF409_DRY_RUN=1 bash head_eval.sh dec0_s20 40000   # print, run nothing
set -uo pipefail

ARM="${1:?usage: head_eval.sh <arm> <stop steps> [head steps]}"
STOP="${2:?usage: head_eval.sh <arm> <stop steps> [head steps]}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
HEAD_STEPS="${3:-$CF409_HEAD_STEPS}"
cf409_require_arm "$ARM" || exit $?
cf409_require_stop "$STOP" || exit $?
cf409_require_head_steps "$HEAD_STEPS" || exit $?

RUNNER="$CF409_PARENT/scripts/head_eval_bb.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no head script at $RUNNER" >&2; exit 2; }

TAG="$(cf409_tag "$ARM" "$STOP" "$HEAD_STEPS")"
BB="$(cf409_bb_ckpt "$ARM" "$STOP")"
BB_GPU="${BB_GPU:-0}"
# One eval root per arm, the same rule as the backbones: two arms in one eval/
# tree cannot be told apart by directory.
ARM_ROOT="$(cf409_arm_root "$ARM")"
mkdir -p "$CF409_RESULTS"

if [ -n "${CF409_DRY_RUN:-}" ]; then
  echo "head $ARM stop=$STOP steps=$HEAD_STEPS seed=$CF409_HEAD_SEED" \
       "enc=$CF409_ENC TAG=$TAG"
  echo "  runner=$RUNNER"
  echo "  bb=${BB:-<not trained yet>}"
  echo "  eval=$(cf409_eval_dir "$ARM" "$TAG")"
  echo "  score=$(cf409_score_file "$ARM" "$STOP")"
  exit 0
fi

[ -n "$BB" ] && [ -f "$BB" ] || {
  echo "ABORT: no bb$(( STOP / 1000 ))k checkpoint for arm $ARM under" \
       "$(cf409_leg_dir "$ARM" "$STOP")" >&2
  exit 3; }

echo "[$(date '+%m-%d %H:%M:%S')] [#409] head $TAG on $(basename "$BB")" \
  | tee -a "$CF409_RESULTS/heads.log"
# CF373_ROOT places the head checkpoint and the eval output under this arm's
# root. CF_RESULTS places the score file in this study's results/. HEAD_SEED is
# the card's, stated rather than inherited, so a change to #373's default
# cannot move this card's heads.
CF373_ROOT="$ARM_ROOT" CF_RESULTS="$CF409_RESULTS" WT="$CF409_WT" \
  HEAD_SEED="$CF409_HEAD_SEED" CF_STOP_K="$(( STOP / 1000 ))" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$TAG" "$BB" "$CF409_ENC" "$HEAD_STEPS"
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] [#409] head $TAG rc=$rc" \
  | tee -a "$CF409_RESULTS/heads.log"
exit $rc
