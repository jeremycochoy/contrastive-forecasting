#!/bin/bash
# #404 — one student head on one arm's backbone, and its 97-config GIFT-Eval.
#
# #373's `head_eval_bb.sh` already takes an EXPLICIT backbone path and a head
# budget, for exactly this reason: a backbone that its own cell table does not
# name. So this wrapper resolves the checkpoint, builds the tag and hands both
# over. The protocol below it is #373's, unchanged — quantile head, 2-layer
# transformer, forecast-len 16, batch 256, lr 1e-3, head seed 20260722, then
# the 97 GIFT-Eval configs under strategy B4 on the CPU.
#
# #401 ran this head protocol in its phase 1, at 30,000 head steps on a bb40k
# backbone. The card asks for the same, so the scores compare.
#
# Usage:  head_eval.sh <arm> <stop steps> [head steps]
#         BB_GPU=0 bash head_eval.sh a08 40000
set -uo pipefail

ARM="${1:?usage: head_eval.sh <arm> <stop steps> [head steps]}"
STOP="${2:?usage: head_eval.sh <arm> <stop steps> [head steps]}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
HEAD_STEPS="${3:-$CF404_HEAD_STEPS}"
cf404_require_arm "$ARM" || exit $?
cf404_require_stop "$STOP" || exit $?
cf404_require_head_steps "$HEAD_STEPS" || exit $?

RUNNER="$CF404_PARENT/scripts/head_eval_bb.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no head script at $RUNNER" >&2; exit 2; }

TAG="$(cf404_tag "$ARM" "$STOP" "$HEAD_STEPS")"
BB="$(cf404_bb_ckpt "$ARM" "$STOP")"
BB_GPU="${BB_GPU:-0}"
# The head's own root, one per arm. The backbones already take one root per
# arm, and the argument is the same for the heads: two arms in one eval/ tree
# cannot be told apart by directory.
ARM_ROOT="$(cf404_arm_root "$ARM")"
mkdir -p "$CF404_RESULTS"

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "head $ARM stop=$STOP steps=$HEAD_STEPS enc=$CF404_ENC TAG=$TAG"
  echo "  runner=$RUNNER"
  echo "  bb=${BB:-<not trained yet>}"
  echo "  eval=$(cf404_eval_dir "$ARM" "$TAG")"
  echo "  score=$CF404_RESULTS/score_${TAG}.txt"
  exit 0
fi

[ -n "$BB" ] && [ -f "$BB" ] || {
  echo "ABORT: no bb$(( STOP / 1000 ))k checkpoint for arm $ARM under" \
       "$(cf404_leg_dir "$ARM" "$STOP")" >&2
  exit 3; }

echo "[$(date '+%m-%d %H:%M:%S')] [#404] head $TAG on $(basename "$BB")" \
  | tee -a "$CF404_RESULTS/heads.log"
# CF373_ROOT places the head checkpoint and the eval output under this arm's
# root; CF_RESULTS places the score file and the stop log in this study's
# results/.
CF373_ROOT="$ARM_ROOT" CF_RESULTS="$CF404_RESULTS" WT="$CF404_WT" \
  CF_STOP_K="$(( STOP / 1000 ))" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$TAG" "$BB" "$CF404_ENC" "$HEAD_STEPS"
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] [#404] head $TAG rc=$rc" \
  | tee -a "$CF404_RESULTS/heads.log"
exit $rc
