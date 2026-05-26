#!/bin/bash
# #320 — forked-continuation arms with a 6-LAYER FORECASTER.
# The data-side counterpart to #318's loss-side 6L-forecaster follow-up.
#
# Each arm is byte-identical to the corresponding #318 1L-forecaster forked arm
# EXCEPT --num-layers 6 (forecaster stack 1→6; the 6L causal *encoder*
# --num-encoder-layers 6 is unchanged). The fork is in the DATA: one (or a
# fraction of) forked-ARIMA pair(s) injected per 256-batch (--synth-kind
# forked-arma --mix-ratio MIX), identical past → divergent futures, so position
# cannot encode the future. Everything else byte-for-#318.
#
# Usage: train_backbone_forked_6Lf.sh <arm> <mix> <tag> [gpu] [chunk]
#   arm    beta | alltime        (base contrastive loss)
#   mix     mix-ratio, e.g. 0.0078125 (=2/256), 0.10, 0.5
#   tag     run-name fragment, e.g. forked2 | forked10pct | forked
#   gpu     CUDA_VISIBLE_DEVICES (default 0)
#   chunk   XSHH_ALLT_CHUNK for the all-time Gram (default 8; alltime only)
set -uo pipefail
ARM="${1:?arm = beta|alltime}"; MIX="${2:?mix-ratio}"; TAG="${3:?tag}"
GPU="${4:-0}"; export XSHH_ALLT_CHUNK="${5:-8}"
SEED=20260520
case "$ARM" in
  beta)    SHAPE=cosine_similarity_batch_full_hh_negs;      NAME="bb_beta_${TAG}_6Lf_50k" ;;
  alltime) SHAPE=cosine_similarity_batch_full_hh_negs_xshh_allt; NAME="bb_xshh_allt_${TAG}_6Lf_50k" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-26_forked_6Lf
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [6Lf-fork $ARM $TAG] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"; export CUDA_VISIBLE_DEVICES="$GPU"
log "BB START shape=$SHAPE num-layers=6(forecaster) forked-arma mix=$MIX chunk=$XSHH_ALLT_CHUNK GPU=$GPU -> 50000"
python3 -u "$TRAIN" \
  --batch-size 256 --device cuda --total-steps 50000 --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
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
rc=$?; [ $rc -ne 0 ] && log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
