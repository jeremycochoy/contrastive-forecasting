#!/bin/bash
# #344 — CPC InfoNCE auxiliary term on the two stop-grad full-forecaster arms
# from the stopgrad-capacity report (#341):
#   enc3  3-layer encoder + full-width (384/6) 6-layer forecaster   [report arm 2]
#   enc6  6-layer encoder + full-width (384/6) 6-layer forecaster   [report arm 3]
# Each is the EXACT same-arm baseline recipe (allt·0.8% + crossfade triplet,
# qk-norm, attn-out-norm, xshh_allt loss, --pos-in-denominator,
# --subtract-contrastive-floor, --stopgrad-positive-h, tau 0.10, batch 1024,
# 12 500 steps, seed 20260520) with ONE addition: --cpc-infonce-weight 1.0 —
# the van den Oord 2018 (Eq. 4, k=1) CPC InfoNCE term through a new learnable
# log-bilinear W_1, summed equal-weight. The CPC term uses NO stop-grad; the
# +sg still governs only the existing contrastive term. Single GPU per arm.
#
#   train_backbone_cpc.sh <arm: enc3|enc6> <gpu> [steps] [save_every]
set -uo pipefail
ARM="${1:?arm (enc3|enc6)}"; GPU="${2:?gpu}"; STEPS="${3:-12500}"; SAVE_EVERY="${4:-2500}"
SEED=20260520
WT="${WT:-/home/jupyter/contrastive-forecasting/.claude/worktrees/exp+cpc-infonce-344}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
case "$ARM" in
  # enc3 (report arm 2): smaller encoder. The added CPC term pushes the
  # baseline's chunk-2/no-grad-ckpt recipe to ~96% of 24 GB, so use the same
  # memory-safe knobs as enc6 (forecaster grad-ckpt + Gram chunk 1 + CPC
  # cross-batch chunk 64); all are byte-identical to the loss (memory ↔ kernel
  # launches only). Gives comfortable headroom for the 12.5k-step run.
  enc3) ENC_LAYERS=6; NENC=3; export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1; export CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}" ;;
  # enc6 (report arm 3): largest config — like its #341 baseline it needs the
  # forecaster gradient-checkpointed and the all-time Gram at chunk 1 to fit a
  # 24 GB card; CPC cross-batch at 64 (smoke-verified to fit alongside).
  enc6) ENC_LAYERS=6; NENC=6; export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1; export CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}" ;;
  *) echo "unknown arm: $ARM (want enc3|enc6)"; exit 2 ;;
esac
NAME="bb_allt08_xftrip_nobn_${ARM}_sgpos_qk_aon_b1024_cpc"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
# GRU patch-encoder memory management for 1-GPU batch 1024 (byte-identical;
# chunks + gradient-checkpoints the GRU). Same settings as the baseline arms.
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-cpc-$ARM g$GPU] $*"; }
[ -n "$HF_TOKEN" ] || { log "WARN: empty HF_TOKEN — HF stream will throttle the GPU"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
# Crash resilience: resume from the latest periodic checkpoint (model+opt+step+RNG+data).
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START arm=$ARM bs=1024 steps=$STEPS xshh_chunk=$XSHH_ALLT_CHUNK cpc_chunk=$CPC_CB_CHUNK ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --stopgrad-positive-h --cpc-infonce-weight 1.0 \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
