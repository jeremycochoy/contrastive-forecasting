#!/bin/bash
# #371 — train ONE curve head cell (single locus × depth) + full-97 eval.
# Same protocol as downstream_seed2.sh's curve cells: 10k re-adapt from
# the step12500 parent head, 1k warmup. Idempotent; skips if FINAL /
# summary.txt already exist. Use to retry OOM'd cells or fill extended
# loci (30000/35000/37500) after the backbone extension.
#
#   dl_one_cell.sh <head_layers: 2|6> <gpu> <step>
set -uo pipefail
HL="${1:?head_layers}"; GPU="${2:?gpu}"; STEP="${3:?step}"
: "${WT:?}"; : "${EXP:?}"; : "${SYNC:?}"
NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"
RES="$EXP/results"; HEADS="$SYNC/heads"; mkdir -p "$RES" "$HEADS"

# Locate BB checkpoint at STEP — original 0->25k emits _step<N>.pth,
# the resumed 25k->37.5k extension emits _r2_step<N>.pth.
BB=""
for cand in "$SYNC/${NAME}_step${STEP}.pth" "$SYNC/${NAME}_r"*"_step${STEP}.pth"; do
  [ -f "$cand" ] && { BB="$cand"; break; }
done
[ -n "$BB" ] || { echo "ABORT: no bb at step $STEP in $SYNC" >&2; exit 1; }
PARENT_HEAD="$HEADS/qhead_${HL}L_${TAG}_step12500_FINAL.pth"
[ -f "$PARENT_HEAD" ] || { echo "ABORT: parentstep head missing at $PARENT_HEAD" >&2; exit 1; }

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-cell-${HL}L-step${STEP} g$GPU] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }

qn="qhead_${HL}L_${TAG}_step${STEP}"
qf="$HEADS/${qn}_FINAL.pth"
out="$RES/gift_eval_full_${TAG}_step${STEP}_${HL}L"

# Atomic claim so parallel workers on the same cell list don't race.
CLAIM="$HEADS/.claim_${qn}"
if ! mkdir "$CLAIM" 2>/dev/null; then
  if [ -f "$qf" ] && [ -f "$out/summary.txt" ]; then
    log "skip (claimed by another worker, already done)"; exit 0
  fi
  log "skip (claim held by another worker)"; exit 0
fi
trap 'rm -rf "$CLAIM"' EXIT

if [ -f "$qf" ]; then
  log "QH skip (FINAL exists)"
else
  log "QH train 10000 on $(basename "$BB") resume=$(basename "$PARENT_HEAD")"
  python3 -u "$QTRAIN" --resume "$PARENT_HEAD" --backbone-path "$BB" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps 10000 --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$HEADS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH FAILED"; exit 1; }
  if   [ -f "$HEADS/${qn}_final.pth" ]; then cp -f "$HEADS/${qn}_final.pth" "$qf"
  elif [ -f "$HEADS/${qn}_best.pth" ];  then cp -f "$HEADS/${qn}_best.pth"  "$qf"
  else cp -f "$(ls -t "$HEADS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH no checkpoint"; exit 1; }
  log "QH done"
fi

if [ -f "$out/summary.txt" ]; then
  log "EVAL skip GM=$(gm "$out/summary.txt")"
else
  mkdir -p "$out"
  log "EVAL full-97 start"
  python3 -u "$QEVAL" --backbone-path "$BB" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_${TAG}_step${STEP}_${HL}L.log" 2>&1 || { log "EVAL FAILED"; exit 1; }
  log "EVAL done GM=$(gm "$out/summary.txt")"
fi
