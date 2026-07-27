#!/bin/bash
# #374 arm 3 — backbone + full downstream for split_pred_rep with
# --moco-negatives (teacher-in-cross-batch-fh). Identical recipe to arm 1
# otherwise: B=512, 12,500 steps, seed 20260520, same τ / SIGReg λ.
set -uo pipefail
WT=/workspace/cf-374
OUT="$WT/experiments/2026-07-10_split_pred_rep"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
NAME="bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
SEED=20260520
STEPS=12500; SAVE_EVERY=2500
ENC_LAYERS=6; NENC=3
export PYTHONPATH="$WT:/workspace/gift-eval/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export GIFT_EVAL="${GIFT_EVAL:-/workspace/gift-eval-data}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [vast-moco-arm3] $*"; }
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN"; exit 1; }

BB="$RUNS/${NAME}_FINAL.pth"
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; }
tlog="$RES/run_${NAME}.log"

if [ ! -f "$BB" ]; then
  RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
  [ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
  log "BB START (arm 3, moco-negatives on) steps=$STEPS bs=512"
  python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
    --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
    --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_split_pred_rep \
    --moco-negatives \
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
    log "BB train exited rc=$rc"; exit 1
  fi
  if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
  elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
  else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
  [ -f "$BB" ] && log "BB DONE -> ${NAME}_FINAL.pth"
fi

# Downstream — same protocol as arm 1: 2L + 6L q-head best-loss (30k) then
# last-checkpoint (10k re-adapt), full-97 GIFT-Eval per (head, ckpt).
BBLAST="$RUNS/${NAME}_final.pth"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){
  local HL="$1" qn="$2" bb="$3" src="$4" tot="$5" wu="$6"
  local qf="$RUNS/${qn}_FINAL.pth"
  [ -f "$qf" ] && { log "QH $qn skip"; return 0; }
  local rflag=(); [ -n "$src" ] && rflag=(--resume "$src")
  log "QH $qn train $tot"
  python3 -u "$QTRAIN" "${rflag[@]}" --backbone-path "$bb" --forecast-len 16 --quantile-head \
    --head-arch transformer --head-causal true --head-num-layers "$HL" --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f --total-steps "$tot" --batch-size 256 --lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --weight-decay 0.1 --schedule cosine --warmup-steps "$wu" --final-lr-ratio 0.1 \
    --save-every 100000 --log-every 200 --save-dir "$RUNS" --run-name "$qn" \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --device cuda "${arch[@]}" --mix-ratio 0.0 --reconstruction forecaster --amp-dtype none \
    >>"$RES/run_${qn}.log" 2>&1 || { log "QH $qn FAILED"; return 1; }
  if   [ -f "$RUNS/${qn}_best.pth" ];  then cp -f "$RUNS/${qn}_best.pth"  "$qf"
  elif [ -f "$RUNS/${qn}_final.pth" ]; then cp -f "$RUNS/${qn}_final.pth" "$qf"
  fi
  [ -f "$qf" ] && log "QH $qn done"; }

do_eval(){
  local HL="$1" qn="$2" bb="$3" tag="$4"
  local qf="$RUNS/${qn}_FINAL.pth"
  local out="$RES/gift_eval_full_${tag}_${HL}L"
  [ -f "$out/summary.txt" ] && { log "EVAL ${tag}_${HL}L skip"; return 0; }
  mkdir -p "$out"; local resume_flag=""
  [ -f "$out/all_results.csv" ] && resume_flag="--resume"
  log "EVAL ${tag}_${HL}L start${resume_flag:+ (resuming)}"
  python3 -u "$QEVAL" $resume_flag --backbone-path "$bb" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_${tag}_${HL}L.log" 2>&1 || { log "EVAL ${tag}_${HL}L FAILED"; return 1; }
  log "EVAL ${tag}_${HL}L done"; }

TAG_ARM3="split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
[ -f "$BBLAST" ] || { log "ABORT: last-ckpt backbone missing at $BBLAST"; exit 1; }
for HL in 2 6; do
  qn_best="qhead_${HL}L_${TAG_ARM3}"
  qn_last="qhead_${HL}L_${TAG_ARM3}_last"
  train_head "$HL" "$qn_best" "$BB"     ""                                "30000" "2000" || exit 1
  do_eval    "$HL" "$qn_best" "$BB"     "$TAG_ARM3"                                       || exit 1
  train_head "$HL" "$qn_last" "$BBLAST" "$RUNS/${qn_best}_FINAL.pth"      "10000" "1000" || exit 1
  do_eval    "$HL" "$qn_last" "$BBLAST" "${TAG_ARM3}_last"                                || exit 1
done
log "arm 3 complete for both HL=2 and HL=6"
