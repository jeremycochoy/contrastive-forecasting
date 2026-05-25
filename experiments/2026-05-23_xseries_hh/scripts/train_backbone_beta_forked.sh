#!/bin/bash
# #318 follow-up — forked-continuation ARIMA on the BASE β loss (PR #319 request).
# = the β recipe (cosine_similarity_batch_full_hh_negs — keeps the adjacent xy
#   term) + ONE forked-ARIMA pair (2 samples) per 256-row batch (--mix-ratio
#   2/256), the other 254 rows real. Isolates whether the fork helps the
#   strongest baseline (β) directly. Differs from train_backbone_forked.sh ONLY
#   in --loss-shape (β vs …_xshh_allt); everything else byte-for-β.
#
# Usage: train_backbone_beta_forked.sh [gpu] [mix_ratio]
set -uo pipefail
GPU="${1:-0}"; MIX="${2:-0.0078125}"   # 2/256 = 1 forked pair/batch
SEED=20260520
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/cross-series-hh
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-23_xseries_hh
NAME="bb_beta_forked2_50k"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [beta-forked] $*"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"; export CUDA_VISIBLE_DEVICES="$GPU"
log "BB START β-loss + forked-arma mix=$MIX GPU=$GPU -> 50000"
python3 -u "$TRAIN" \
  --batch-size 256 --device cuda --total-steps 50000 --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
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
