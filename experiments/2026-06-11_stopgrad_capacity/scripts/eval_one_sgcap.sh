#!/bin/bash
# #341 — one GIFT-Eval invocation for a sgcap cell, local elisa artifacts (no
# sync mirror: backbone + head both live in $EXP/runs). Same eval call as
# downstream_sgcap.sh::do_eval; supports the shard/mop-up drivers via
# EVAL_OUT_OVERRIDE + EVAL_CONFIG_FILTER (raw '<ds>/<term>' names).
#   eval_one_sgcap.sh <head_run_name> <bb_file> <out_tag> <head_layers> <gpu>
set -uo pipefail
QN="${1:?head_run_name}"; BBF="${2:?bb_file}"; OUT_TAG="${3:?out_tag}"; HL="${4:?head_layers}"; GPU="${5:?gpu}"
WT="${WT:-/tmp/cf-341}"
EXP="${EXP:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity}"
RUNS="$EXP/runs"; RES="$EXP/results"; mkdir -p "$RES"
BB="$RUNS/$BBF"
HEAD="$RUNS/${QN}_FINAL.pth"; [ -f "$HEAD" ] || HEAD="$RUNS/${QN}_best.pth"
out="${EVAL_OUT_OVERRIDE:-$RES/gift_eval_full_${OUT_TAG}_${HL}L}"
FILTER=(); [ -n "${EVAL_CONFIG_FILTER:-}" ] && FILTER=(--config-filter "$EVAL_CONFIG_FILTER")
[ -f "$out/summary.txt" ] && { echo "EVAL $OUT_TAG ${HL}L skip (summary exists)"; exit 0; }
[ -f "$BB" ] || { echo "ABORT missing backbone $BB"; exit 1; }
[ -f "$HEAD" ] || { echo "ABORT missing head $HEAD"; exit 1; }
mkdir -p "$out"
# Bottleneck backbones need the forecaster width passed explicitly (the eval
# rebuilds full-width by default); full-width + encoder depth auto-detect.
FCST=(); case "$BBF" in *_bn_enc6_*) FCST=(--forecaster-d-model 128 --forecaster-n-heads 4) ;; esac
export PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=8
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
echo "[$(date '+%m-%d %H:%M:%S')] EVAL $OUT_TAG ${HL}L start (head=$(basename "$HEAD"), gpu=$GPU)"
elog="$RES/run_eval_full_${OUT_TAG}_${HL}L.log"
[ -n "${EVAL_OUT_OVERRIDE:-}" ] && elog="$out/run_eval.log"
python3 -u "$QEVAL" --resume "${FILTER[@]}" --backbone-path "$BB" --head-path "$HEAD" --output-dir "$out" --strategy B4 \
  --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
  "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
  >>"$elog" 2>&1 || { echo "EVAL $OUT_TAG ${HL}L FAILED"; exit 1; }
echo "[$(date '+%m-%d %H:%M:%S')] EVAL $OUT_TAG ${HL}L done: $(grep 'Aggregate GM-Relative MASE' "$out/summary.txt" 2>/dev/null || echo 'sharded (no aggregate)')"
