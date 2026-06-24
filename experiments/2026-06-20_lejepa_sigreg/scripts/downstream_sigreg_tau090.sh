#!/bin/bash
# #357 free-arm — downstream q-head + GIFT-Eval for the τ=0.9 EMA arm.
# Same head-matched protocol as the τ=0.98 / no-EMA arms.
#
#   downstream_sigreg_tau090.sh <head_layers: 2|6> <gpu>
set -uo pipefail
HL="${1:?head_layers}"; GPU="${2:?gpu}"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
WT="${WT:-/tmp/contrastive-forecasting-357}"
OUT="${OUT:-/tmp/contrastive-forecasting-357/reports/2026-06-21_lejepa_sigreg_tau098}"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/bb_${TAG}_FINAL.pth"; BBLAST="$RUNS/bb_${TAG}_final.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="$GPU"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl-sigreg-tau090 ${HL}L g$GPU] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }
[ -f "$BB" ] || { log "ABORT backbone missing: $BB"; exit 1; }
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){
  local qn="$1" bb="$2" src="$3" tot="$4" wu="$5" qf="$RUNS/$1_FINAL.pth"
  [ -f "$qf" ] && { log "QH $qn skip (FINAL exists)"; return 0; }
  local rflag=(); [ -n "$src" ] && rflag=(--resume "$src")
  log "QH $qn train $tot on $(basename "$bb")"
  python3 -u "$QTRAIN" "${rflag[@]}" --backbone-path "$bb" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps "$tot" --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps "$wu" --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH $qn FAILED (tail: $(tail -3 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }; log "QH $qn done"; }

do_eval(){
  local qf="$RUNS/$1_FINAL.pth" out="$RES/gift_eval_full_$3_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL $3 skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; log "EVAL $3 full-97 start"
  python3 -u "$QEVAL" --backbone-path "$2" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_$3_${HL}L.log" 2>&1 || { log "EVAL $3 FAILED"; return 1; }
  log "EVAL $3 done GM=$(gm "$out/summary.txt")"; }

train_head "qhead_${HL}L_${TAG}" "$BB" "" 30000 2000 \
  && { [ "${DO_EVAL:-1}" = 1 ] && do_eval "qhead_${HL}L_${TAG}" "$BB" "$TAG" || true; }
if [ -f "$BBLAST" ]; then
  train_head "qhead_${HL}L_${TAG}_last" "$BBLAST" "$RUNS/qhead_${HL}L_${TAG}_FINAL.pth" 10000 1000 \
    && { [ "${DO_EVAL:-1}" = 1 ] && do_eval "qhead_${HL}L_${TAG}_last" "$BBLAST" "${TAG}_last" || true; }
else log "no final.pth ($BBLAST) — skipping last head"; fi
log "sigreg-tau090 downstream complete (${HL}L)"
