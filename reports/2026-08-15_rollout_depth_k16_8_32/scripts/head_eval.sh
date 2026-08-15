#!/bin/bash
# #401 — one student head on one backbone stop, and its 97-config GIFT-Eval.
#
# #373's `head_eval_bb.sh` already takes an EXPLICIT backbone path and a head
# budget, for exactly this reason: a backbone that its own cell table does
# not name. So this wrapper resolves the checkpoint, builds the tag and hands
# both over. The protocol below it is #373's, unchanged — quantile head,
# 2-layer transformer, forecast-len 16, batch 256, lr 1e-3, head seed
# 20260722, then the 97 GIFT-Eval configs under strategy B4 on the CPU.
#
# The head budget is the one thing that varies:
#   phase 1   30,000 steps on every stop
#   phase 2   the stop's own step count, 40k / 100k / 200k
# Both write their own tag, so phase 2 never reads phase 1's score file.
#
# Usage:  head_eval.sh <k> <stop steps> [head steps]
#         BB_GPU=0 bash head_eval.sh 16 40000            # phase 1
#         BB_GPU=0 bash head_eval.sh 16 40000 40000      # phase 2
set -uo pipefail

K="${1:?usage: head_eval.sh <k> <stop steps> [head steps]}"
STOP="${2:?usage: head_eval.sh <k> <stop steps> [head steps]}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
HEAD_STEPS="${3:-$CF401_HEAD_STEPS_P1}"
cf401_require_depth "$K" || exit $?
cf401_require_stop "$STOP" || exit $?
cf401_require_head_steps "$HEAD_STEPS" "$STOP" || exit $?

RUNNER="$CF401_PARENT/scripts/head_eval_bb.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no head script at $RUNNER" >&2; exit 2; }

TAG="$(cf401_tag "$K" "$STOP" "$HEAD_STEPS")"
BB="$(cf401_bb_ckpt "$K" "$STOP")"
BB_GPU="${BB_GPU:-0}"
# The head's own root, one per depth. The backbones already take one root
# per depth (see cf401_arm_root), and the argument is the same for the
# heads: a later glob that forgets the depth would otherwise resolve to
# another arm's head, and three arms in one eval/ tree cannot be told apart
# by directory.
ARM_ROOT="$(cf401_arm_root "$K")"
mkdir -p "$CF401_RESULTS"

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "head k=$K stop=$STOP HEAD_STEPS=$HEAD_STEPS enc=$CF401_ENC TAG=$TAG"
  echo "  runner=$RUNNER"
  echo "  bb=${BB:-<not trained yet>}"
  echo "  root=$ARM_ROOT"
  echo "  eval=$(cf401_eval_dir "$K" "$TAG")"
  echo "  score=$CF401_RESULTS/score_${TAG}.txt"
  exit 0
fi

[ -n "$BB" ] && [ -f "$BB" ] || {
  echo "ABORT: no bb$(( STOP / 1000 ))k checkpoint for k=$K under $(cf401_leg_dir "$K" "$STOP")" >&2
  exit 3; }

echo "[$(date '+%m-%d %H:%M:%S')] [#401] head $TAG on $(basename "$BB")" \
  | tee -a "$CF401_RESULTS/heads.log"
# CF373_ROOT places the head checkpoint and the eval output under this
# arm's root; CF_STUDY_DIR places the score file and the log in this
# study's results/.
CF373_ROOT="$ARM_ROOT" CF_STUDY_DIR="$CF401_STUDY" WT="$CF401_WT" \
  CF_RESULTS="$CF401_RESULTS" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$TAG" "$BB" "$CF401_ENC" "$HEAD_STEPS"
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] [#401] head $TAG rc=$rc" \
  | tee -a "$CF401_RESULTS/heads.log"
exit $rc
