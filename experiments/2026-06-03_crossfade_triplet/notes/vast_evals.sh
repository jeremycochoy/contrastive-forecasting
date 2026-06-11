#!/bin/bash
# Re-run only the 4 GIFT-Eval full-97 evals (q-heads already trained). Bottleneck-128 arch.
set -u
TAG=allt08_xftrip_bn_enc6_qk_aon_b1024
WT=/root/cf-328; RUNS=/root/out/runs; RES=/root/out/results
export PYTHONPATH="$WT:/root/gift_eval_pkg" GIFT_EVAL=/root/gift-eval-data
export HF_TOKEN=$(cat $WT/experiments/hf_token.txt) HUGGING_FACE_HUB_TOKEN=$(cat $WT/experiments/hf_token.txt)
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [evals] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }
run(){ # head_layers  ckpt_suffix  backbone
  local HL=$1 sfx=$2 bb=$3 qf out
  qf="$RUNS/qhead_${HL}L_${TAG}${sfx}_FINAL.pth"; out="$RES/gift_eval_full_${TAG}${sfx}_${HL}L"
  [ -f "$out/summary.txt" ] && { say "${HL}L${sfx} skip GM=$(gm "$out/summary.txt")"; return; }
  rm -rf "$out"; mkdir -p "$out"; say "${HL}L${sfx} eval start"
  python3 -u "$QEVAL" --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru --rev-norm-kind ewma \
    --rev-norm-span 128 --device cuda --head-causal true \
    > "$RES/run_eval_${TAG}${sfx}_${HL}L.log" 2>&1 && say "${HL}L${sfx} done GM=$(gm "$out/summary.txt")" || say "${HL}L${sfx} FAILED: $(tail -2 "$RES/run_eval_${TAG}${sfx}_${HL}L.log"|tr '\n' ' ')"
}
run 2 ""     "$RUNS/bb_${TAG}_FINAL.pth"
run 6 ""     "$RUNS/bb_${TAG}_FINAL.pth"
run 2 "_last" "$RUNS/bb_${TAG}_final.pth"
run 6 "_last" "$RUNS/bb_${TAG}_final.pth"
say "ALL EVALS DONE"
