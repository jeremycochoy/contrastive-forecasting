#!/bin/bash
# #371 — B=512 seed2 downstream: 2L and 6L quantile heads + full-97
# GIFT-Eval at loci {12500, 15000, 20000, 25000}. Protocol mirrors
# #369's dl_at_step.sh so the curves are comparable.
#
# Per depth:
#   step12500 — fresh 30k head (2k warmup), 12.5k = seed-control replica
#   step{15000,20000,25000} — resume 10k re-adapt from the 12.5k head (1k warmup)
#
#   downstream_seed2.sh <head_layers: 2|6> <gpu>
set -uo pipefail
HL="${1:?head_layers}"; GPU="${2:?gpu}"
: "${WT:?}"; : "${EXP:?}"; : "${SYNC:?}"
NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"
RES="$EXP/results"; mkdir -p "$RES"
HEADS="$SYNC/heads"; mkdir -p "$HEADS"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-seed2-${HL}L g$GPU] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }

train_head(){ # step total warmup resume_src
  local step="$1" total="$2" warmup="$3" src="$4"
  local bb="$SYNC/${NAME}_step${step}.pth"
  local qn="qhead_${HL}L_${TAG}_step${step}"
  local qf="$HEADS/${qn}_FINAL.pth"
  [ -f "$bb" ] || { log "ABORT: bb $bb missing"; return 1; }
  [ -f "$qf" ] && { log "QH $qn skip (FINAL exists)"; return 0; }
  local rflag=(); [ -n "$src" ] && [ -f "$src" ] && rflag=(--resume "$src")
  log "QH $qn train $total on $(basename "$bb") $([ -n "$src" ] && echo resume=$(basename "$src"))"
  python3 -u "$QTRAIN" "${rflag[@]}" --backbone-path "$bb" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps "$total" --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps "$warmup" --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$HEADS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH $qn FAILED"; return 1; }
  if   [ -f "$HEADS/${qn}_final.pth" ]; then cp -f "$HEADS/${qn}_final.pth" "$qf"
  elif [ -f "$HEADS/${qn}_best.pth" ];  then cp -f "$HEADS/${qn}_best.pth"  "$qf"
  else cp -f "$(ls -t "$HEADS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }
  log "QH $qn done"
}

do_eval(){ # step
  local step="$1"
  local bb="$SYNC/${NAME}_step${step}.pth"
  local qf="$HEADS/qhead_${HL}L_${TAG}_step${step}_FINAL.pth"
  local out="$RES/gift_eval_full_${TAG}_step${step}_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL step${step} skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"
  log "EVAL step${step} full-97 start"
  python3 -u "$QEVAL" --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_${TAG}_step${step}_${HL}L.log" 2>&1 || { log "EVAL step${step} FAILED"; return 1; }
  log "EVAL step${step} done GM=$(gm "$out/summary.txt")"
}

fail=0
run_cell(){ # step total warmup resume_src
  train_head "$1" "$2" "$3" "$4" || { fail=$((fail+1)); return; }
  do_eval "$1" || fail=$((fail+1))
}

# Cell 1 — step12500 fresh 30k head (seed-control replica of arm-C `last`).
run_cell 12500 30000 2000 ""
PARENT_HEAD="$HEADS/qhead_${HL}L_${TAG}_step12500_FINAL.pth"
# Curve cells — 10k re-adapt resumed from the 12500 head.
for st in 15000 20000 25000; do
  run_cell "$st" 10000 1000 "$PARENT_HEAD"
done
log "downstream complete (${HL}L): failed cells=$fail"
exit "$fail"
