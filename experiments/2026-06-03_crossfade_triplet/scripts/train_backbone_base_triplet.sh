#!/bin/bash
# #328 disentanglement — base + triplet: the cleanest triplet isolation. Exactly
# the base config (6-layer encoder + 128-wide bottleneck forecaster) with ONLY the
# crossfade triplet added (--crossfade-triplets 1), on the allt-0.8% base. base vs
# base+triplet differ by the triplet alone -> directly attributes the triplet's
# effect on the unmodified architecture. Same recipe/seed/12500 steps as every arm.
#   train_backbone_base_triplet.sh <gpu> [steps] [save_every]
set -uo pipefail
GPU="${1:?gpu}"; STEPS="${2:-12500}"; SAVE_EVERY="${3:-2500}"; SEED=20260520
WT="${WT:-/tmp/cf-328}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet}"
NAME="bb_allt08_xftrip_bn_enc6_qk_aon_b1024"     # bn = 128-wide bottleneck forecaster (the base config)
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK="${PATCH_ENC_CHUNK:-4}"
export XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-1}"   # 6L encoder + all-time Gram: chunk 1 to fit one card (byte-identical)
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [base+trip g$GPU] $*"; }
[ -f "$BB" ] && { log "SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START enc=6 bottleneck=128 triplet=1 bs=1024 steps=$STEPS ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
[ $rc -ne 0 ] && { log "BB exited rc=$rc — NOT creating FINAL (--resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"; exit 1; }
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
