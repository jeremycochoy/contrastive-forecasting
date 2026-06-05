#!/bin/bash
# #328 disentanglement: single-change ablation arms on the 0.8%-fork base (NO crossfade).
# Same stabilised recipe (qk-norm + attn-out-norm + pos-in-denominator +
# subtract-contrastive-floor), batch 1024, 12500 steps, seed 20260520. Each arm
# changes only architecture vs the base (6-layer encoder + 128-wide forecaster):
#   L3       : --num-encoder-layers 3, keep bottleneck
#   L3_nobn  : --num-encoder-layers 3, drop bottleneck (forecaster = encoder width)
#   nobn     : --num-encoder-layers 6, drop bottleneck
#
#   train_backbone_ablation.sh <arm> <gpu> [steps] [save_every]
set -uo pipefail
ARM="${1:?arm=L3|L3_nobn|nobn}"; GPU="${2:?gpu}"; STEPS="${3:-12500}"; SAVE_EVERY="${4:-2500}"
SEED=20260520
case "$ARM" in
  L3)      ENC=3; FCST=(--forecaster-d-model 128 --forecaster-n-heads 4); TAG=allt08_L3_qk_aon_b1024 ;;
  L3_nobn) ENC=3; FCST=();                                                TAG=allt08_L3_nobn_qk_aon_b1024 ;;
  nobn)    ENC=6; FCST=();                                                TAG=allt08_nobn_qk_aon_b1024 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
WT="${WT:-/tmp/cf-328}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet}"
NAME="bb_${TAG}"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-2}" CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4   # GRU memory mgmt for 1-GPU b1024 (byte-identical)
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-$ARM g$GPU] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START enc=$ENC bottleneck=${FCST[*]:-none} bs=1024 steps=$STEPS chunk=$XSHH_ALLT_CHUNK ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers "$ENC" --num-layers 6 "${FCST[@]}" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
# Primary FINAL = best-loss checkpoint (the project convention; "best"). The
# full-training (last, step 12500) checkpoint is also kept as final.pth and
# evaluated separately via eval_last_ablation.sh (the 3-layer arms drift up in
# loss late, so best vs full-training is reported for both).
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
# Keep an explicit full-training copy so downstream can evaluate it too.
[ -f "$RUNS/${NAME}_final.pth" ] && cp -f "$RUNS/${NAME}_final.pth" "$RUNS/${NAME}_LAST.pth" 2>/dev/null || true
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
