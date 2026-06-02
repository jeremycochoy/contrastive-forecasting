#!/bin/bash
# #322 — downstream for the batch-1024 6L-forecaster backbones. IDENTICAL to #320's
# downstream_6Lf.sh (same q-head recipe + GIFT-Eval) so b256↔b1024 is a clean paired
# comparison — ONLY WT/OUT are repointed to this card. The q-head batch stays 256;
# #322 varies only the backbone's contrastive batch. --num-layers 6 (6L forecaster)
# in both q-head training and eval; q-head depth (2|6) set by <head_layers>.
# --amp-dtype none avoids the depthwise-conv fp32/bf16 mismatch.
#
#   downstream_b1024.sh <backbone_path> <head_layers> <tag> <gpu> <evalsets>
set -uo pipefail
BB="${1:?backbone}"; HL="${2:?head_layers}"; TAG="${3:?tag}"; GPU="${4:?gpu}"; SETS="${5:-triage full}"
FCST=(--forecaster-d-model 128 --forecaster-n-heads 4)
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
QN="qhead_${HL}L_${TAG}"; QF="$RUNS/${QN}_FINAL.pth"
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [b1024 ${TAG} ${HL}L g$GPU] $*"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
[ -f "$BB" ] || { log "ABORT backbone missing: $BB"; exit 1; }
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)
if [ -f "$QF" ]; then log "QH skip (FINAL exists)"; else
  log "QH train 30k (${HL}L head) on $(basename "$BB")"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QTRAIN" \
    --backbone-path "$BB" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$QN" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${QN}.log" 2>&1 || { log "QH FAILED (tail: $(tail -3 "$RES/run_${QN}.log"|tr '\n' ' '))"; exit 1; }
  if   [ -f "$RUNS/${QN}_best.pth" ];  then cp -f "$RUNS/${QN}_best.pth"  "$QF"
  elif [ -f "$RUNS/${QN}_final.pth" ]; then cp -f "$RUNS/${QN}_final.pth" "$QF"
  else cp -f "$(ls -t "$RUNS/${QN}"_*k.pth 2>/dev/null|head -1)" "$QF"; fi
  [ -f "$QF" ] || { log "QH FAILED no checkpoint"; exit 1; }; log "QH done"
fi
do_eval(){ local set="$1" filt="$2" out="$RES/gift_eval_${1}_${TAG}_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL $set skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt"); log "EVAL $set start"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$QEVAL" \
    --backbone-path "$BB" --head-path "$QF" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 6 "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_${set}_${TAG}_${HL}L.log" 2>&1 || { log "EVAL $set FAILED"; return 1; }
  log "EVAL $set done GM=$(gm "$out/summary.txt")"; }
for s in $SETS; do case "$s" in
  triage) do_eval triage "$TRIAGE" || true ;; full) do_eval full "" || true ;; esac; done
find "$RUNS" -maxdepth 1 -name "${QN}_*.pth" ! -name "${QN}_FINAL.pth" -delete 2>/dev/null
log "cell complete (pruned ${QN} intermediates)"
