#!/bin/bash
# Run q-head (30k 2L-causal) + GIFT-Eval (triage 11 + full 97) LOCALLY on
# elisa's free 4090 from a SYNCED backbone checkpoint. Recipe is
# byte-identical to #303 run_all.sh downstream() (the of-record q-head).
# Free compute (no vast spend) + fast (4090 = of-record card). Idempotent.
#   local_downstream.sh <arm> <gpu>
set -uo pipefail
WT=/home/jupyter/cf-wt-crossed-loss
SY=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation
ARM="${1:?arm}"; GPU="${2:?gpu}"
NAME="cl_${ARM}_50k"
SRC="$SY/sync_${ARM}"
DST="$SY/downstream_${ARM}"; RUNS="$DST/runs"; RES="$DST/results"
mkdir -p "$RUNS" "$RES"
# backbone: prefer FINAL, else best_loss (resume-capable, same weights of record)
BB=""
for c in "${NAME}_FINAL" "${NAME}_best_loss" "${NAME}_best_gap"; do
  [ -f "$SRC/runs/${c}.pth" ] && { BB="$SRC/runs/${c}.pth"; break; }
done
[ -n "$BB" ] || { echo "[$ARM] NO synced backbone in $SRC/runs"; exit 1; }
QN="${NAME}_qhead_xfmr2L_quant_30k"; QF="$RUNS/${NAME}_qhead_FINAL.pth"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$ARM g$GPU] $*"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null|head -1|grep -oE '[0-9]+\.[0-9]+'|head -1; }

log "backbone=$BB"
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
      --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
      --rev-norm-kind ewma --rev-norm-span 128)
if [ ! -f "$QF" ]; then
  log "QH START 30k"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${QN}.log" 2>&1 || { log "QH FAILED (tail: $(tail -3 "$RES/run_${QN}.log"|tr '\n' ' '))"; exit 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
fi
[ -f "$QF" ] || { log "QH no checkpoint"; exit 1; }
log "QH ready -> $QF"
do_eval(){ # tag outdir filter
  local tag="$1" out="$2" filt="$3"
  [ -f "$out/summary.txt" ] && { log "EVAL $tag exists GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  log "EVAL $tag START"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 1 --forecaster-d-model 128 --forecaster-n-heads 4 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_${tag}.log" 2>&1 || { log "EVAL $tag FAILED (tail: $(tail -3 "$RES/run_eval_${tag}.log"|tr '\n' ' '))"; return 1; }
  log "EVAL $tag DONE GM=$(gm "$out/summary.txt")"
}
do_eval "triage_${ARM}" "$RES/gift_eval_triage_${NAME}" "$TRIAGE" || true
do_eval "full_${ARM}"   "$RES/gift_eval_full_${NAME}"   ""        || true
log "DOWNSTREAM COMPLETE triageGM=$(gm "$RES/gift_eval_triage_${NAME}/summary.txt") fullGM=$(gm "$RES/gift_eval_full_${NAME}/summary.txt")"
