#!/bin/bash
# #322 — forked arms × 6L forecaster at GLOBAL BATCH 1024, SINGLE-GPU.
# Pivot from 2-GPU DDP: this shared box's GPU 0 is permanently held by foreign
# 9–14-day Jupyter kernels (~8 GB), so DDP (needs both cards near-empty) is
# infeasible. Instead we run the full 1024-batch on ONE card (GPU 1). A single
# process @1024 pools ALL 1024 in the negatives natively (no gather needed) — the
# card's requirement, satisfied trivially. The GRU patch-encoder is gradient-
# checkpointed + chunked (PATCH_ENC_CKPT/CHUNK) so 1024 fits in ~12 GB allocated
# (measured) instead of OOM-ing. Checkpointing is byte-identical in the forward, so
# the trained backbone equals #320's recipe at 4× batch. Everything else is
# byte-identical to #320.
#
# Usage: train_backbone_b1024_1gpu.sh <arm> <mix> <tag> <steps> [save_every] [allt_chunk] [gpu]
set -uo pipefail
ARM="${1:?arm=beta|alltime}"; MIX="${2:?mix}"; TAG="${3:?tag}"; STEPS="${4:?steps}"
SAVE_EVERY="${5:-2500}"; export XSHH_ALLT_CHUNK="${6:-2}"; GPU="${7:-1}"
SEED=20260520
case "$ARM" in
  beta)    SHAPE=cosine_similarity_batch_full_hh_negs;            NAME="bb_beta_${TAG}_6Lf_b1024" ;;
  alltime) SHAPE=cosine_similarity_batch_full_hh_negs_xshh_allt;  NAME="bb_xshh_allt_${TAG}_6Lf_b1024" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [b1024-1gpu $ARM $TAG] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START shape=$SHAPE batch=1024 (1-GPU, GRU-ckpt) forked-arma mix=$MIX allt_chunk=$XSHH_ALLT_CHUNK steps=$STEPS gpu=$GPU ${RESUME}"
python3 -u "$TRAIN" $RESUME \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape "$SHAPE" --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio "$MIX" \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  log "BB FAILED rc=$rc"; exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
if [ -f "$BB" ]; then
  find "$RUNS" -maxdepth 1 -name "${NAME}_*.pth" ! -name "${NAME}_FINAL.pth" -delete 2>/dev/null
  log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1)); pruned intermediates"
  exit 0
fi
log "BB FAILED no checkpoint"; exit 1
