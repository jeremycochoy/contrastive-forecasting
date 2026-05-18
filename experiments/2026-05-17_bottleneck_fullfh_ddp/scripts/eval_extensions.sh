#!/bin/bash
# Evaluate one milestone backbone: 2L causal-transformer q-head (30k, v13
# recipe — identical to the 50k pipeline) -> GIFT-Eval triage (11) then
# full (97) -> GM-Relative MASE. Idempotent (skip if summary/FINAL
# exists), single-instance per TAG. Single-GPU; run two of these in
# parallel pinned to different GPUs to evaluate 100k and 150k at once.
#   usage: eval_extensions.sh <backbone_FINAL.pth> <TAG> <GPU_ID>
set -uo pipefail
BB="$1"; TAG="$2"; GPU="$3"; QSTEPS="${4:-30000}"   # QSTEPS: q-head total-steps (default 30k recipe)
EXP=/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp
CODE=/home/jupyter/cf-wt-bottleneck-fullfh
QTRAIN="$CODE/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$CODE/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
RUNS="$EXP/runs"; RES="$EXP/results"
HFT=/home/jupyter/contrastive-forecasting/experiments/hf_token.txt
LOG="$RES/eval_ext_${TAG}.log"; ST="$RES/eval_ext_${TAG}.status"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
export PYTHONPATH="$CODE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="$(cat "$HFT")" HUGGING_FACE_HUB_TOKEN="$(cat "$HFT")"
export GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$CODE"
exec 9>"$RES/.eval_${TAG}.lock"
flock -n 9 || { echo "[lock] eval_extensions $TAG already running — exit"; exit 0; }
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [$TAG] $*" | tee -a "$LOG"; }
gm(){ grep -aE 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
[ -f "$BB" ] || { log "backbone missing: $BB"; echo "FAILED missing-backbone" >"$ST"; exit 1; }
echo "RUNNING $(date '+%F %T')" >"$ST"
log "=== eval $TAG START — backbone=$(basename "$BB") GPU=$GPU ==="

ARCH=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 \
      --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
      --rev-norm-kind ewma --rev-norm-span 128)

QN="${TAG}_qhead_xfmr2L_quant_$((QSTEPS/1000))k"; QF="$RUNS/${QN}_FINAL.pth"
if [ ! -f "$QF" ]; then
  log "q-head train ${QSTEPS} -> $QN"
  python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps "$QSTEPS" --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 --save-every 5000 --log-every 200 \
    --save-dir "$RUNS" --run-name "$QN" --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${ARCH[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${QN}.log" 2>&1 || { log "q-head FAILED"; echo "FAILED qhead" >"$ST"; exit 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
fi
log "q-head ready"

do_eval(){ # $1=label $2=outdir $3=filter|''
  local lbl="$1" out="$2" filt="$3"
  [ -f "$out/summary.txt" ] && { log "$lbl exists GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  log "$lbl eval starting"
  python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 1 \
    --forecaster-d-model 128 --forecaster-n-heads 4 --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_${TAG}_${lbl}.log" 2>&1 || { log "$lbl eval FAILED"; return 1; }
  log "$lbl eval DONE GM=$(gm "$out/summary.txt")"
}
do_eval triage "$RES/gift_eval_triage_${TAG}" "$TRIAGE" || true
do_eval full   "$RES/gift_eval_full_${TAG}"   ""        || true
TG=$(gm "$RES/gift_eval_triage_${TAG}/summary.txt"); FG=$(gm "$RES/gift_eval_full_${TAG}/summary.txt")
log "=== eval $TAG DONE — triage=${TG:-?} full=${FG:-?} ==="
echo "DONE triage=${TG:-?} full=${FG:-?} $(date '+%F %T')" >"$ST"
