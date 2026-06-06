#!/bin/bash
# Full-training (last-checkpoint) eval (#328 reviewer request: also evaluate the
# step-12500 backbone, on top of the best-loss eval). Resumes the best-loss-trained
# q-head on the arm's final.pth backbone, re-adapts briefly (warm start), then runs
# GIFT-Eval full-97. Cheap vs a fresh 30k head. Output tag: <TAG>_last.
#   eval_last_ablation.sh <arm> <head_layers> <gpu> [resume_steps]
#   arm ∈ L3 | L3_nobn | nobn | xftrip(#328)
set -uo pipefail
ARM="${1:?arm}"; HL="${2:?head_layers}"; GPU="${3:?gpu}"; STEPS="${4:-10000}"
case "$ARM" in
  L3)      FCST=(--forecaster-d-model 128 --forecaster-n-heads 4); TAG=allt08_L3_qk_aon_b1024 ;;
  L3_nobn) FCST=();                                                TAG=allt08_L3_nobn_qk_aon_b1024 ;;
  nobn)    FCST=();                                                TAG=allt08_nobn_qk_aon_b1024 ;;
  xftrip)  FCST=();                                                TAG=allt08_xftrip_nobn_enc3_qk_aon_b1024 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
WT="${WT:-/tmp/cf-328}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet}"
RUNS="$OUT/runs"; RES="$OUT/results"
BBLAST="$RUNS/bb_${TAG}_final.pth"               # last checkpoint (step 12500)
SRC="$RUNS/qhead_${HL}L_${TAG}_FINAL.pth"        # best-loss-trained head to resume
QN="qhead_${HL}L_${TAG}_last"; QF="$RUNS/${QN}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [last-$ARM ${HL}L g$GPU] $*"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
[ -f "$BBLAST" ] || { log "ABORT no final.pth: $BBLAST"; exit 1; }
[ -f "$SRC" ]    || { log "ABORT no best-loss head: $SRC"; exit 1; }
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)
if [ -f "$QF" ]; then log "QH(last) skip (FINAL exists)"; else
  log "QH(last) resume $(basename "$SRC") on final.pth, re-adapt ${STEPS} steps"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QTRAIN" --resume "$SRC" \
    --backbone-path "$BBLAST" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps "$STEPS" --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${QN}.log" 2>&1 || { log "QH(last) FAILED (tail: $(tail -3 "$RES/run_${QN}.log"|tr '\n' ' '))"; exit 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
  [ -f "$QF" ] || { log "QH(last) FAILED no checkpoint"; exit 1; }; log "QH(last) done"
fi
out="$RES/gift_eval_full_${TAG}_last_${HL}L"
if [ -f "$out/summary.txt" ]; then log "EVAL(last) skip GM=$(gm "$out/summary.txt")"; else
  mkdir -p "$out"; log "EVAL(last) full-97 start on final.pth"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QEVAL" \
    --backbone-path "$BBLAST" --head-path "$QF" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 6 "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true \
    >>"$RES/run_eval_full_${TAG}_last_${HL}L.log" 2>&1 || { log "EVAL(last) FAILED"; exit 1; }
  log "EVAL(last) done GM=$(gm "$out/summary.txt")"
fi
