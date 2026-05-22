#!/bin/bash
# #313 follow-up (PR #315 review item 3) — does a STRONGER head change the
# align-vs-(B) story? Train a 6-layer transformer quantile q-head on BOTH the
# (B)+align+floor backbone and the (B) backbone (existing checkpoints), then
# GIFT-Eval triage(11) + full(97) each. Only the head depth changes vs the
# 2L runs (2L→6L); everything else identical, so 2L vs 6L is clean.
#
# Two backbones × {head train, full eval} run one-per-GPU in parallel.
#   followup_6L.sh
set -uo pipefail
WT=/home/jupyter/cf-wt-align-floor
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B
CLABL=/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation
RUNS=$MAIN/runs; RES=$MAIN/results; mkdir -p "$RUNS" "$RES"
FCST=(--forecaster-d-model 128 --forecaster-n-heads 4)   # both backbones are (B)-bneck
TRIAGE='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
LG="$RES/followup_6L.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [6L] $*" | tee -a "$LG"; }
gm(){ grep -E 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }

arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 1 "${FCST[@]}" \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

# train_6L <tag> <backbone_path> <gpu>  -> writes $RUNS/<tag>_qhead6L_FINAL.pth
train_6L(){
  local tag=$1 bb=$2 g=$3
  local qn=${tag}_qhead_xfmr6L_quant_30k qf=$RUNS/${tag}_qhead6L_FINAL.pth
  [ -f "$qf" ] && { log "$tag 6L head exists -> skip"; return 0; }
  [ -f "$bb" ] || { log "$tag ABORT: backbone $bb missing"; return 1; }
  log "$tag 6L head TRAIN START 30k on GPU$g"
  CUDA_VISIBLE_DEVICES="$g" python3 -u "$QTRAIN" \
    --backbone-path "$bb" --forecast-len 16 --quantile-head --head-arch transformer \
    --head-causal true --head-num-layers 6 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 --lr 1e-3 --beta1 0.9 --beta2 0.98 \
    --weight-decay 0.1 --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype bf16 \
    >>"$RES/run_${qn}.log" 2>&1 || { log "$tag 6L head FAILED (tail: $(tail -3 "$RES/run_${qn}.log"|tr '\n' ' '))"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  else cp -f "$(ls -t "$RUNS/${qn}"_*k.pth 2>/dev/null|head -1)" "$qf"; fi
  [ -f "$qf" ] && { log "$tag 6L head DONE -> $(basename "$qf")"; return 0; }
  log "$tag 6L head FAILED no checkpoint"; return 1
}

# eval_bb <tag> <backbone_path> <qf> <gpu> <outtag> <filter>
eval_bb(){
  local tag=$1 bb=$2 qf=$3 g=$4 out="$RES/gift_eval_$5" filt=$6
  [ -f "$out/summary.txt" ] && { log "$5 exists GM=$(gm "$out/summary.txt")"; return 0; }
  [ -f "$qf" ] || { log "$5 SKIP (no head)"; return 1; }
  mkdir -p "$out"; local ff=(); [ -n "$filt" ] && ff=(--config-filter "$filt")
  CUDA_VISIBLE_DEVICES="$g" python3 -u "$QEVAL" \
    --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 \
    --num-layers 1 "${FCST[@]}" --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 \
    --device cuda --head-causal true "${ff[@]}" \
    >>"$RES/run_eval_$5.log" 2>&1 || { log "$5 eval FAILED (tail: $(tail -3 "$RES/run_eval_$5.log"|tr '\n' ' '))"; return 1; }
  log "$5 DONE GM=$(gm "$out/summary.txt")"
}

ALIGN_BB="$RUNS/bb_alignfloor_50k_FINAL.pth"
B_BB="$CLABL/runs/cl_hh_50k_FINAL.pth"

log "=== PHASE 1: train 6L heads (align@GPU1 ∥ B@GPU0) ==="
train_6L bb_alignfloor_50k "$ALIGN_BB" 1 & pa=$!
train_6L B_cl_hh_50k       "$B_BB"     0 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
log "phase1 done rc(align)=$ra rc(B)=$rb"
AQF="$RUNS/bb_alignfloor_50k_qhead6L_FINAL.pth"; BQF="$RUNS/B_cl_hh_50k_qhead6L_FINAL.pth"

log "=== PHASE 2: full-97 eval (align@GPU1 ∥ B@GPU0) ==="
eval_bb align "$ALIGN_BB" "$AQF" 1 full_bb_alignfloor_50k_6L "" & ea=$!
eval_bb B     "$B_BB"     "$BQF" 0 full_B_cl_hh_50k_6L       "" & eb=$!
wait $ea; wait $eb

log "=== PHASE 3: triage(11) eval ==="
eval_bb align "$ALIGN_BB" "$AQF" 1 triage_bb_alignfloor_50k_6L "$TRIAGE" & ta=$!
eval_bb B     "$B_BB"     "$BQF" 0 triage_B_cl_hh_50k_6L       "$TRIAGE" & tb=$!
wait $ta; wait $tb

AF=$(gm "$RES/gift_eval_full_bb_alignfloor_50k_6L/summary.txt"); BF=$(gm "$RES/gift_eval_full_B_cl_hh_50k_6L/summary.txt")
AT=$(gm "$RES/gift_eval_triage_bb_alignfloor_50k_6L/summary.txt"); BT=$(gm "$RES/gift_eval_triage_B_cl_hh_50k_6L/summary.txt")
echo "FOLLOWUP_6L_DONE $(date -u +%FT%TZ) align_full=$AF align_triage=$AT B_full=$BF B_triage=$BT" > "$RES/.followup6L_done"
log "=== COMPLETE — 6L heads ===
  (B)+align+floor : full=$AF  triage=$AT
  (B) baseline    : full=$BF  triage=$BT
  [2L ref: align full 1.4308/triage 1.6154 ; B full 1.3572/triage 1.4461]"
