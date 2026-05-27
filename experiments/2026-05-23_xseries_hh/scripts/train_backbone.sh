#!/bin/bash
# #318 — train the "β + cross-series same-step h↔h" backbone on elisa.
#
# BYTE-IDENTICAL to the #309 β arm recipe (elisa_run.sh beta tau01):
#   bottleneck d=128/h=4, AdamW β2=0.98, τ=0.10, fp16 body / fp32
#   residual+patch-emb, dropkey 0.70 shared, GRU patch-enc, ewma span128,
#   seed 20260520, 50k, global batch 256, --pos-in-denominator.
# The ONLY change: --loss-shape cosine_similarity_batch_full_hh_negs_xshh
#   (= full_hh_negs, drop adjacent log_neg_xy, add cross-series same-step
#    h↔h at every step). See src/loss.py and #318.
#
# Code from the WORKTREE; checkpoints to the MAIN checkout (CLAUDE.md:
# valuable untracked state never lives in an auxiliary worktree).
#
# Usage: train_backbone.sh [gpu] [total_steps]
#   gpu          CUDA_VISIBLE_DEVICES (default "0"; GPU1 is often busy)
#   total_steps  default 50000 (use a small value for a smoke test)
set -uo pipefail
GPU="${1:-0}"
TOTAL="${2:-50000}"
SEED=20260520

WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh
NAME="bb_xshh_50k"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [xshh] $*"; }

[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
log "BB START loss=xshh τ=0.10 β2=0.98 bneck d=128 fp16 1-GPU bs256 GPU=$GPU -> $TOTAL steps"
python3 -u "$TRAIN" \
  --batch-size 256 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
# Promote the converged checkpoint to a stable FINAL name (Checkpoint Safety Rule 1).
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
for c in "${NAME}_best_loss" "${NAME}_final"; do
  [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
done
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
