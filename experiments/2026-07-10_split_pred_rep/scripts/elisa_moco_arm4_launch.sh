#!/bin/bash
# #374 arm 4 — backbone + full downstream on elisa for the CHAMPION shape
# (arm C's cosine_similarity_batch_full_hh_negs_xshh_allt) with
# --moco-negatives (teacher-in-cross-batch-fh). Identical recipe to arm C
# (#366 λ_e=1, λ_h=1, EMA τ=0.90, --pos-in-denominator
# --subtract-contrastive-floor) otherwise: B=512, 12,500 steps,
# seed 20260520, contrastive τ=0.10, same synth / freq / mixup config.
# Backbone on GPU 1; downstream 2L on GPU 0 + 6L on GPU 1 in parallel
# (same train_head / do_eval protocol as downstream_sigreg.sh: best-loss
# q-head 30k + last-checkpoint q-head 10k re-adapt, full-97 GIFT-Eval B4
# per (head, ckpt)).
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-374}"
OUT="$WT/experiments/2026-07-10_split_pred_rep"
RUNS="$OUT/runs_arm4"; RES="$OUT/results_arm4"; mkdir -p "$RUNS" "$RES"
NAME="bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090"
TAG_ARM4="${NAME#bb_}"
SEED=20260520
STEPS=12500; SAVE_EVERY=2500
ENC_LAYERS=6; NENC=3
BB_GPU="${BB_GPU:-1}"; GPU_2L="${GPU_2L:-0}"; GPU_6L="${GPU_6L:-1}"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
QTRAIN="$WT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
QEVAL="$WT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [elisa-moco-arm4] $*"; }
gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }
[ -f "$TRAIN" ]  || { log "ABORT: TRAIN script not found at $TRAIN (check WT=$WT)"; exit 2; }
[ -f "$QTRAIN" ] || { log "ABORT: QTRAIN script not found at $QTRAIN"; exit 2; }
[ -f "$QEVAL" ]  || { log "ABORT: QEVAL script not found at $QEVAL"; exit 2; }
[ -f "$HF_TOKEN_PATH" ] || { log "ABORT: HF token not found at $HF_TOKEN_PATH — HF stream throttles GPU, refusing to launch"; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN — HF stream throttles GPU, refusing to launch"; exit 2; }

BB="$RUNS/${NAME}_FINAL.pth"
BBLAST="$RUNS/${NAME}_final.pth"
tlog="$RES/run_${NAME}.log"

if [ -f "$BB" ]; then
  log "BB SKIP ($NAME FINAL exists)"
else
  RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
  [ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
  log "BB START (arm 4: xshh_allt + moco-negatives) gpu=$BB_GPU steps=$STEPS bs=512 ${RESUME}"
  CUDA_VISIBLE_DEVICES="$BB_GPU" python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
    --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
    --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
    --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
    --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
    --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
    --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
    --depthwise-conv 3 --deprecated-depthwise-conv 0 \
    --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
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
    log "BB train exited rc=$rc — NOT creating FINAL. tail: $(tail -3 "$tlog"|tr '\n' ' ')"
    exit 1
  fi
  if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
  elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
  else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
  [ -f "$BB" ] || { log "BB FAILED no checkpoint"; exit 1; }
  log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"
fi

# Downstream — same protocol as downstream_sigreg.sh: best-loss q-head
# (30k) + last-checkpoint q-head (resumed, 10k re-adapt) per head_layers,
# full-97 GIFT-Eval B4 per cell. 2L and 6L run in parallel on their GPUs.
arch=(--t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 --num-layers 6 \
      --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128)

train_head(){ # HL gpu run_name backbone resume_src total warmup
  local HL="$1" gpu="$2" qn="$3" bb="$4" src="$5" tot="$6" wu="$7" qf="$RUNS/$3_FINAL.pth"
  [ -f "$qf" ] && { log "QH $qn skip (FINAL exists)"; return 0; }
  local rflag=(); [ -n "$src" ] && rflag=(--resume "$src")
  log "QH $qn train $tot on $(basename "$bb") (gpu $gpu)"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$QTRAIN" "${rflag[@]}" --backbone-path "$bb" --forecast-len 16 --quantile-head \
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

do_eval(){ # HL gpu run_name backbone out_tag
  local HL="$1" gpu="$2" qf="$RUNS/$3_FINAL.pth" out="$RES/gift_eval_full_$5_${1}L"
  [ -f "$out/summary.txt" ] && { log "EVAL $5 ${HL}L skip GM=$(gm "$out/summary.txt")"; return 0; }
  mkdir -p "$out"; log "EVAL $5 ${HL}L full-97 start (gpu $gpu)"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u "$QEVAL" --backbone-path "$4" --head-path "$qf" --output-dir "$out" --strategy B4 \
    --forecast-len 16 --t-raw 4096 --backbone-c 1 --d-model 384 --n-heads 6 --num-layers 6 \
    --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128 --device cuda --head-causal true \
    >>"$RES/run_eval_full_$5_${HL}L.log" 2>&1 || { log "EVAL $5 ${HL}L FAILED"; return 1; }
  log "EVAL $5 ${HL}L done GM=$(gm "$out/summary.txt")"; }

downstream_hl(){ # HL gpu — runs both (head, ckpt) cells; returns failed-cell count
  local HL="$1" gpu="$2" fail=0
  local qn_best="qhead_${HL}L_${TAG_ARM4}" qn_last="qhead_${HL}L_${TAG_ARM4}_last"
  if train_head "$HL" "$gpu" "$qn_best" "$BB" "" 30000 2000; then
    do_eval "$HL" "$gpu" "$qn_best" "$BB" "$TAG_ARM4" || fail=$((fail+1))
  else fail=$((fail+1)); fi
  if [ ! -f "$BBLAST" ]; then
    log "ABORT ${HL}L: last-checkpoint backbone missing at $BBLAST (train.py"
    log "    emits \${run-name}_final.pth at end of training; absence means"
    log "    BB did not complete — the last cells depend on it)."
    return $((fail+1))
  fi
  if train_head "$HL" "$gpu" "$qn_last" "$BBLAST" "$RUNS/${qn_best}_FINAL.pth" 10000 1000; then
    do_eval "$HL" "$gpu" "$qn_last" "$BBLAST" "${TAG_ARM4}_last" || fail=$((fail+1))
  else fail=$((fail+1)); fi
  return "$fail"
}

log "downstream start: 2L on GPU $GPU_2L + 6L on GPU $GPU_6L (parallel)"
downstream_hl 2 "$GPU_2L" >>"$RES/dl_2L.log" 2>&1 &
pid2=$!
downstream_hl 6 "$GPU_6L" >>"$RES/dl_6L.log" 2>&1 &
pid6=$!
log "downstream PIDs: 2L=$pid2  6L=$pid6"
wait $pid2; rc2=$?
wait $pid6; rc6=$?
log "arm 4 complete: 2L failed-cells=$rc2  6L failed-cells=$rc6"
exit $(( rc2 != 0 || rc6 != 0 ))
