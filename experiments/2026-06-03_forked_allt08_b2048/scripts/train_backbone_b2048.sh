#!/bin/bash
# #327 — allt·0.8% forked arm retrained at GLOBAL BATCH 2048, SINGLE-GPU on GPU 1.
# Follow-up to #322 (allt·0.8% at batch 1024 = 2L 1.213 / 6L 1.198). Same stabilised
# recipe — qk-norm + attn-out-norm (the 4x pool collapses without them; the 8x pool
# needs them at least as much), --subtract-contrastive-floor + --pos-in-denominator,
# LR 1e-3, tau 0.10, seed 20260520 — with the contrastive batch doubled 1024 -> 2048,
# all 2048 terms pooled in the negatives natively (one process, no gather).
#
# WHY single-GPU @2048 (not DDP): a single process @2048 OOMs the backbone-TRANSFORMER
# forward at ~22 GB; 2-GPU DDP @1024/rank fits the forward but its 2048-pooled loss needs
# ~19 GB/rank, which OOMs on GPU 0 (permanently ~4.5 GB foreign-held) — #322's "GPU 0 is
# occupied -> pivot to single-GPU" situation, one batch-doubling later. The fix that fits
# 2048 on the free GPU 1: BACKBONE_CKPT gradient-checkpoints the backbone transformer's
# non-fp32 layers (new env flag in src/blocks.py, mirroring PATCH_ENC_CKPT for the GRU).
# Checkpointing is BYTE-IDENTICAL (validated: an 8-step run with the flag off vs on gives
# bit-identical loss + gap), so the trained backbone equals the recipe at 2x batch.
# Measured: ~20.5 GB peak, fwd 6.4 s + bwd 7.7 s = ~14 s/step -> ~24.5 h / 6250 steps.
#
# Step budget 6250 = same 12.8M samples seen as #322 (6250*2048 == 12500*1024), isolating
# the negative-pool size (1024 -> 2048) from training length — #322's clean-isolation logic.
#
# Usage: train_backbone_b2048.sh <steps> <save_every> <allt_chunk> <gpu> [run_tag]
set -uo pipefail
STEPS="${1:-6250}"; SAVE_EVERY="${2:-1250}"; export XSHH_ALLT_CHUNK="${3:-4}"; GPU="${4:-1}"
TAG="${5:-forked2_qk_aon}"
LR="${LR:-1e-3}"
SEED=20260520
NAME="bb_xshh_allt_${TAG}_6Lf_b2048"
SHAPE=cosine_similarity_batch_full_hh_negs_xshh_allt

WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-allt08-b2048
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_forked_allt08_b2048
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK="${PATCH_ENC_CHUNK:-8}" BACKBONE_CKPT=1
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [b2048 $TAG] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START shape=$SHAPE batch=2048 (1-GPU, GRU-ckpt chunk=$PATCH_ENC_CHUNK + BACKBONE_CKPT) lr=$LR qk+aon mix=0.0078125 allt_chunk=$XSHH_ALLT_CHUNK steps=$STEPS gpu=$GPU ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 2048 --device cuda --total-steps "$STEPS" --lr "$LR" --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 50 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape "$SHAPE" --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  log "BB FAILED rc=$rc"; exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|grep -v optimizer|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1)); kept all checkpoints"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
