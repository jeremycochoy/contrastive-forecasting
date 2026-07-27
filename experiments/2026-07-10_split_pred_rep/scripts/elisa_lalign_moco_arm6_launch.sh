#!/bin/bash
# #374 arm 6 — L_align (MoCo) + L_rep on elisa.
#
# Same recipe as arm 5, only difference: the alignment term is MoCo-style
# InfoNCE (student encoder as anchor, EMA teacher as keys, cross-batch
# same-time negatives) instead of BYOL:
#   --align-loss-weight        0     (arm 5 had 1.0)
#   --align-moco-loss-weight   1.0   (new for arm 6)
# `L_rep` (h-anchored families, no positive) stays unchanged from arm 5.
set -uo pipefail
GPU="${GPU:-1}"
STEPS="${STEPS:-12500}"; SAVE_EVERY="${SAVE_EVERY:-2500}"
SEED="${SEED:-20260520}"
WT=/tmp/contrastive-forecasting-374
OUT="$WT/experiments/2026-07-10_split_pred_rep"
ENC_LAYERS=6; NENC=3
TAG_ARM6="lalignmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6_tau090"
NAME="bb_${TAG_ARM6}"
RUNS="$OUT/runs_arm6"; RES="$OUT/results_arm6"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"; BBLAST="$RUNS/${NAME}_final.pth"
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
DL_LOG="$RES/dl_arm6.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [elisa-lalign-moco-arm6] $*" | tee -a "$DL_LOG"; }
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 1; }

# ---- BACKBONE ----
tlog="$RES/run_${NAME}.log"
if [ -f "$BB" ]; then
  log "BB SKIP ($NAME FINAL exists)"
else
  RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
  [ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
  log "BB START arm=lalign-moco bs=512 steps=$STEPS ${RESUME}"
  python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
    --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
    --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_rep_only \
    --align-moco-loss-weight 1.0 \
    --ema-embedding --ema-encoder --ema-tau 0.9 --cpc-infonce-weight 1.0 \
    --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
    --sigreg-embedding-weight 1.0 --sigreg-encoding-weight 1.0 \
    --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
    --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
    --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
    --log-attn-amplitude --log-attn-amplitude-every 200 \
    --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
    >>"$tlog" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "BB train exited rc=$rc — NOT creating FINAL. tail: $(tail -3 "$tlog"|tr '\n' ' ')"
    exit 1
  fi
  if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
  elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
  else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
  [ -f "$BB" ] || { log "BB FAILED no checkpoint"; exit 1; }
  log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"
fi

# ---- DOWNSTREAM (sequential 6L → 2L) ----
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
gm(){ awk '/Aggregate GM-Relative MASE/ { print $NF }' "$1"; }
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){
  local HL="$1" qn="$2" bb="$3" src="$4" tot="$5" wu="$6"
  local qf="$RUNS/${qn}_FINAL.pth"
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
  [ -f "$qf" ] || { log "QH $qn no checkpoint"; return 1; }
  log "QH $qn done"; }
do_eval(){
  local HL="$1" qn="$2" bb="$3" tag="$4"
  local qf="$RUNS/${qn}_FINAL.pth" out="$RES/gift_eval_full_${tag}_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL $tag ${HL}L skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; log "EVAL $tag ${HL}L full-97 start"
  python3 -u "$QEVAL" --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_${tag}_${HL}L.log" 2>&1 || { log "EVAL $tag ${HL}L FAILED"; return 1; }
  log "EVAL $tag ${HL}L done GM=$(gm "$out/summary.txt")"; }
downstream_hl(){ local HL="$1" fail=0
  local qn_best="qhead_${HL}L_${TAG_ARM6}" qn_last="qhead_${HL}L_${TAG_ARM6}_last"
  train_head "$HL" "$qn_best" "$BB" "" 30000 3000 || { fail=$((fail+1)); return $fail; }
  do_eval    "$HL" "$qn_best" "$BB" "$TAG_ARM6"  || fail=$((fail+1))
  [ -f "$BBLAST" ] || { log "ABORT ${HL}L: BBLAST missing"; return $((fail+1)); }
  train_head "$HL" "$qn_last" "$BBLAST" "$RUNS/${qn_best}_FINAL.pth" 10000 1000 || { fail=$((fail+1)); return $fail; }
  do_eval    "$HL" "$qn_last" "$BBLAST" "${TAG_ARM6}_last" || fail=$((fail+1))
  return $fail; }
log "arm6 downstream start (6L → 2L sequential)"
downstream_hl 6 || log "downstream 6L had failures rc=$?"
downstream_hl 2 || log "downstream 2L had failures rc=$?"
log "arm6 ALL DONE"
