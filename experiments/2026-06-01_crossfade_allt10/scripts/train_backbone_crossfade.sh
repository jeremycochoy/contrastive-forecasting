#!/bin/bash
# #325 — allt·10% + 10% regime-crossfade backbone, GLOBAL BATCH 1024, SINGLE-GPU.
# Byte-identical to #322's allt·10% FINAL recipe (single process @1024, GRU patch-
# encoder gradient-checkpointed + chunked, lr 1e-3, qk-norm + attn-out-norm, allt
# loss, forked-arma mix 0.10) EXCEPT it adds --crossfade-ratio 0.10, so the batch is
# 80% real / 10% forked-arma / 10% crossfade (#325). The crossfade rows are blended
# from the real sub-batch (no extra HF rows). The two attention norms are the #322
# b1024 collapse fix and lr 1e-3 is the #322 FINAL (the 5e-4 default in #322's
# generic launcher was a pre-norms band-aid).
#
# Usage: train_backbone_crossfade.sh [steps] [save_every] [allt_chunk] [gpu]
set -uo pipefail
STEPS="${1:-12500}"; SAVE_EVERY="${2:-2500}"; export XSHH_ALLT_CHUNK="${3:-2}"; GPU="${4:-1}"
LR="${LR:-1e-3}"; MIX="${MIX:-0.10}"; CROSS="${CROSS:-0.10}"; SEED="${SEED:-20260520}"
SHAPE=cosine_similarity_batch_full_hh_negs_xshh_allt
NAME="bb_xshh_allt_forked10pct_crossfade10pct_qk_aon_6Lf_b1024"

WT="${WT:-/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/crossfade-allt10}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10}"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [xfade $NAME] $*"; }
[ -f "$BB" ] && { log "BB SKIP (FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START batch=1024 (1-GPU, GRU-ckpt) lr=$LR qk+aon mix=$MIX crossfade=$CROSS allt_chunk=$XSHH_ALLT_CHUNK steps=$STEPS gpu=$GPU ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr "$LR" --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape "$SHAPE" --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio "$MIX" --crossfade-ratio "$CROSS" \
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
if [ -f "$BB" ]; then log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1)); kept all checkpoints"; exit 0; fi
log "BB FAILED no checkpoint"; exit 1
