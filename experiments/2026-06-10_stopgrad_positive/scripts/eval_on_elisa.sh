#!/bin/bash
# #339 budget restructure — GIFT-Eval full-97 on elisa (inference only; head/
# backbone TRAINING stays on vast.ai). Same eval invocation as
# downstream_sgpos.sh::do_eval, fed from the sync mirror of the vast run.
#   eval_on_elisa.sh <head_run_name> <bb_file> <out_tag> <head_layers> <gpu>
# e.g. eval_on_elisa.sh qhead_2L_<TAG> bb_<TAG>_FINAL.pth <TAG> 2 1
set -uo pipefail
QN="${1:?head_run_name}"; BBF="${2:?bb_file}"; OUT_TAG="${3:?out_tag}"; HL="${4:?head_layers}"; GPU="${5:?gpu}"
WT="${WT:-/tmp/cf-sgpos}"
SYNC="${SYNC:-/home/jupyter/contrastive-forecasting/sync_sgpos_339/runs}"
EXP="${EXP:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive}"
RES="$EXP/results"; mkdir -p "$RES"
BB="$SYNC/$BBF"
# Remote copies <run>_best.pth → <run>_FINAL.pth; the sync mirror carries _best.
HEAD="$SYNC/${QN}_best.pth"; [ -f "$HEAD" ] || HEAD="$SYNC/${QN}_FINAL.pth"
out="${EVAL_OUT_OVERRIDE:-$RES/gift_eval_full_${OUT_TAG}_${HL}L}"
FILTER=(); [ -n "${EVAL_CONFIG_FILTER:-}" ] && FILTER=(--config-filter "$EVAL_CONFIG_FILTER")
[ -f "$out/summary.txt" ] && { echo "EVAL $OUT_TAG ${HL}L skip (summary exists)"; exit 0; }
[ -f "$BB" ] || { echo "ABORT missing backbone $BB"; exit 1; }
[ -f "$HEAD" ] || { echo "ABORT missing head $HEAD"; exit 1; }
mkdir -p "$out"
export PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=8
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
echo "[$(date '+%m-%d %H:%M:%S')] EVAL $OUT_TAG ${HL}L start (head=$(basename "$HEAD"), gpu=$GPU)"
elog="$RES/run_eval_full_${OUT_TAG}_${HL}L.log"
[ -n "${EVAL_OUT_OVERRIDE:-}" ] && elog="$out/run_eval.log"
python3 -u "$QEVAL" --resume "${FILTER[@]}" --backbone-path "$BB" --head-path "$HEAD" --output-dir "$out" --strategy B4 \
  --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
  --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
  >>"$elog" 2>&1 || { echo "EVAL $OUT_TAG ${HL}L FAILED"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] EVAL $OUT_TAG ${HL}L done: $(grep 'Aggregate GM-Relative MASE' "$out/summary.txt")"
