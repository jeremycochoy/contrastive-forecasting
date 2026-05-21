#!/bin/bash
# #309 fp32 continuation on elisa — resume a diverged arm from its
# pre-divergence best_loss checkpoint and continue to 50k in ALL-fp32
# (fp16/bf16 body diverges for the bottleneck-removed arms). Only the
# dtype flags change vs the original arm recipe.
#
# Code from the WORKTREE (has --resume + hh_negs loss); resume checkpoint
# and outputs in the MAIN checkout (CLAUDE.md: valuable state in main).
#
# Usage:  elisa_continue_fp32.sh <arm> <gpu_id>
#   arm = alpha | gamma
set -uo pipefail
ARM="${1:?arm = alpha|gamma}"; GPU="${2:?gpu_id}"
case "$ARM" in
  alpha) BETA2=0.98 ;;
  gamma) BETA2=0.95 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
SEED=20260520
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-20_bottleneck_beta2_confound
SRC="bb_${ARM}_50k"
NAME="bb_${ARM}_fp32cont_50k"
TOTAL=50000
RUNS="$MAIN/runs"; RES="$MAIN/results"; mkdir -p "$RUNS" "$RES"
RESUME="$RUNS/${SRC}_best_loss.pth"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [${ARM}-fp32cont] $*"; }

[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
[ -f "$RESUME" ] || { log "ERROR: resume checkpoint $RESUME missing"; exit 1; }
[ -f "${RESUME%.pth}_optimizer.pth" ] || { log "ERROR: optimizer companion missing"; exit 1; }

tlog="$RES/run_${NAME}.log"
log "BB START $ARM fp32cont resume=$(basename "$RESUME") β2=$BETA2 ALL-fp32 1-GPU bs256 GPU$GPU -> $TOTAL"
CUDA_VISIBLE_DEVICES="$GPU" python3 -u "$TRAIN" \
  --resume "$RESUME" \
  --batch-size 256 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 "$BETA2" --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 \
  --patch-emb-dtype fp32 >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "BB train exited rc=$rc (tail: $(tail -3 "$tlog"|tr '\n' ' '))"; fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
for c in "${NAME}_best_loss" "${NAME}_final"; do
  [ -f "$RUNS/${c}_optimizer.pth" ] && cp -f "$RUNS/${c}_optimizer.pth" "$RUNS/${NAME}_FINAL_optimizer.pth" && break
done
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
