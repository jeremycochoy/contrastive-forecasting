#!/bin/bash
# #412 — one student head on one arm's backbone, and its 97-config GIFT-Eval.
#
# #373's `head_eval_bb.sh` takes an EXPLICIT backbone path and a head budget,
# so this wrapper resolves the checkpoint, builds the tag and hands both over.
# The protocol below it is #373's, unchanged: quantile head, 2-layer
# transformer, forecast length 16, batch 256, lr 1e-3, head seed 20260722,
# then the 97 GIFT-Eval configs under strategy B4 on the CPU. The parents ran
# that protocol at 30,000 head steps on each stop, so the scores compare.
#
# ---- The one thing this wrapper adds ----------------------------------------
#
# The head trainer and the GIFT-Eval REBUILD the backbone before they load its
# weights, and both take the shape from the command line. At `d_model` 64 they
# cannot load an 11.4M checkpoint. `CF_BB_SHAPE` gives them this card's width,
# through #373's `bb_shape.sh`. `head_eval_bb.sh` starts `eval_local.sh` as a
# child, so one exported value reaches the head and the evaluation together.
#
# The head is wider too, because its input is the backbone latent. So the
# memory gate waits for more free memory than #373's default.
#
# Usage:  head_eval.sh <arm> <stop steps> [head steps]
#         BB_GPU=0 bash head_eval.sh k3_r100_09 40000
set -uo pipefail

ARM="${1:?usage: head_eval.sh <arm> <stop steps> [head steps]}"
STOP="${2:?usage: head_eval.sh <arm> <stop steps> [head steps]}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
HEAD_STEPS="${3:-$CF412_HEAD_STEPS}"
cf412_require_arm "$ARM" || exit $?
cf412_require_stop "$STOP" || exit $?
cf412_require_head_steps "$HEAD_STEPS" || exit $?

RUNNER="$CF412_PARENT/scripts/head_eval_bb.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no head script at $RUNNER" >&2; exit 2; }

TAG="$(cf412_tag "$ARM" "$STOP" "$HEAD_STEPS")"
BB="$(cf412_bb_ckpt "$ARM" "$STOP")"
BB_GPU="${BB_GPU:-0}"
BB_SHAPE_ARGS="$(cf412_bb_shape)"
ARM_ROOT="$(cf412_arm_root "$ARM")"
mkdir -p "$CF412_RESULTS"

if [ -n "${CF412_DRY_RUN:-}" ]; then
  echo "head $ARM stop=$STOP steps=$HEAD_STEPS enc=$CF412_ENC TAG=$TAG"
  echo "  runner=$RUNNER"
  echo "  CF_BB_SHAPE=$BB_SHAPE_ARGS"
  echo "  bb=${BB:-<not trained yet>}"
  echo "  eval=$(cf412_eval_dir "$ARM" "$TAG")"
  echo "  score=$(cf412_score_file "$ARM" "$STOP")"
  exit 0
fi

[ -n "$BB" ] && [ -f "$BB" ] || {
  echo "ABORT: no bb$(( STOP / 1000 ))k checkpoint for arm $ARM under" \
       "$(cf412_leg_dir "$ARM" "$STOP")" >&2
  exit 3; }

echo "[$(date '+%m-%d %H:%M:%S')] [#412] head $TAG on $(basename "$BB")" \
  | tee -a "$CF412_RESULTS/heads.log"
# CF373_ROOT places the head checkpoint and the evaluation output under this
# arm's root. CF_RESULTS places the score file and the stop log in this
# study's results/. CF_STOP_K labels the stop in those logs.
CF373_ROOT="$ARM_ROOT" CF_RESULTS="$CF412_RESULTS" WT="$CF412_WT" \
  CF_BB_SHAPE="$BB_SHAPE_ARGS" CF_STOP_K="$(( STOP / 1000 ))" \
  HEAD_VRAM_MIB="$CF412_HEAD_VRAM_MIB" BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$TAG" "$BB" "$CF412_ENC" "$HEAD_STEPS"
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] [#412] head $TAG rc=$rc" \
  | tee -a "$CF412_RESULTS/heads.log"
exit $rc
