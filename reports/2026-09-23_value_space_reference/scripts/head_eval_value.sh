#!/bin/bash
# #415 — one quantile head on one stop, and its 97-config GIFT-Eval under B4
# and under A2.
#
# Usage:  head_eval_value.sh <stop steps> [head steps]
#         BB_GPU=0 bash head_eval_value.sh 40000
#
# The scoring path is #414's, and this script only resolves the checkpoint and
# hands it over. #373's `head_eval_bb.sh` then runs the protocol both cards
# share: a quantile head, 2-layer transformer, forecast length 16, batch 256,
# lr 1e-3, head seed 20260722, 30,000 steps on the FROZEN backbone, then the
# 97 GIFT-Eval configs under strategy B4 on the CPU. Only the backbone
# differs, which is the whole point of the comparison.
#
# B4 rolls the forecast out in LATENT space, and this model trains a VALUE
# space rollout. So `eval_local.sh` then scores the SAME head under A2, which
# feeds the median forecast back as values. The report says which of the two
# numbers answers the card.
#
# The pretraining VALUE HEAD plays no part here. `prepare_backbone_state_dict`
# strips `value_head.*` the way it strips `cpc_w1.*` and `teacher_*`, so the
# head trainer and the eval build the cell's backbone and load it strictly.
# The head they score is trained from scratch, exactly as on every #414 stop.
#
# `CF_BB_SHAPE` gives both of them this card's width. `head_eval_bb.sh` starts
# `eval_local.sh` as a child, so one exported value reaches the head and the
# evaluation together.
#
# Each stop keeps its own `all_results.csv` under
# `<runs>/value_space/eval/<tag>/gift/` for B4 and `.../gift_a2/` for A2,
# which is what the per-dataset comparison against #414's `cos200k` reads.
# Both halves are idempotent: a strategy whose score file exists scores
# nothing.
set -uo pipefail

STOP="${1:?usage: head_eval_value.sh <stop steps> [head steps]}"

. "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
HEAD_STEPS="${2:-$CF415_HEAD_STEPS}"
cf415_is_stop "$STOP" || exit $?

RUNNER="$CF415_PARENT/scripts/head_eval_bb.sh"
EVAL="$CF415_PARENT/scripts/eval_local.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no head script at $RUNNER" >&2; exit 2; }
[ -f "$EVAL" ] || { echo "ABORT: no eval script at $EVAL" >&2; exit 2; }

. "$CF415_PARENT/scripts/leg_paths.sh"
RUNS="$CF415_RUNS" ROOT="$(runs_root)" || exit 2
CELL_RUNS="$ROOT/value_space"
TAG="$(cf415_tag "$STOP")"
OUT="$CELL_RUNS/eval/$TAG"
BB="$(ckpt_at_step "$CELL_RUNS" "$CF415_RUN_NAME" "$(( STOP / 1000 ))")"
# `head_eval_bb.sh` trains the head under this name, and A2 reads it back.
HEAD_CKPT="$OUT/qhead_${TAG}_s${HEAD_SEED:-20260722}_final.pth"
SCORE_B4="$(cf415_score_file "$STOP" B4)"
SCORE_A2="$(cf415_score_file "$STOP" A2)"
BB_GPU="${BB_GPU:-0}"
# A2 doubles the CPU time of a stop. CF415_SCORE_A2=0 scores B4 alone; the
# head stays on disk, so A2 can score it later.
STRATEGIES="B4 A2"
[ "${CF415_SCORE_A2:-1}" = "0" ] && STRATEGIES="B4"
# The default of `head_eval_bb.sh`, so the A2 eval reads the data B4 reads.
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
mkdir -p "$CF415_RESULTS"

if [ -n "${CF415_DRY_RUN:-}" ]; then
  echo "head stop=$STOP steps=$HEAD_STEPS enc=$CF415_ENC TAG=$TAG"
  echo "  runner=$RUNNER"
  echo "  CF_BB_SHAPE=$(cf415_bb_shape)"
  echo "  bb=${BB:-<not trained yet>}"
  echo "  eval=$OUT strategies=$STRATEGIES"
  echo "  score=$SCORE_B4"
  echo "  score_a2=$SCORE_A2"
  exit 0
fi

[ -n "$BB" ] && [ -f "$BB" ] || {
  echo "ABORT: no bb$(( STOP / 1000 ))k checkpoint under $CELL_RUNS" >&2
  exit 3; }

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#415] $*" \
  | tee -a "$CF415_RESULTS/heads.log"; }

log "head $TAG on $(basename "$BB")"
CF373_ROOT="$CELL_RUNS" CF_RESULTS="$CF415_RESULTS" WT="$CF415_WT" \
  CF_BB_SHAPE="$(cf415_bb_shape)" CF_STOP_K="$(( STOP / 1000 ))" \
  HEAD_VRAM_MIB="$CF415_HEAD_VRAM_MIB" BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$TAG" "$BB" "$CF415_ENC" "$HEAD_STEPS"
rc=$?
log "head $TAG + B4 rc=$rc"
[ $rc -eq 0 ] || exit $rc
[ "$STRATEGIES" = "B4" ] && { log "A2 $TAG skipped: CF415_SCORE_A2=0"; exit 0; }

EVAL_STRATEGY=A2 WT="$CF415_WT" CF_BB_SHAPE="$(cf415_bb_shape)" \
  bash "$EVAL" "$TAG" "$(( STOP / 1000 ))" "$CF415_ENC" "$BB" \
  "$HEAD_CKPT" "$OUT" "$SCORE_A2" >>"$OUT/stop.log" 2>&1
rc=$?
log "A2 $TAG rc=$rc"
exit $rc
