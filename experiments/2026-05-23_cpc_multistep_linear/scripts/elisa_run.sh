#!/bin/bash
# #316 — CPC-style multi-step linear forecast on β.
#
# Recipe = the β arm of #309 (the (B) recipe + AdamW β2=0.98, τ=0.10,
# dropkey 0.70 shared, fp16 body / fp32 residual+pemb, seed-controlled, 50k,
# global batch 256) with ONE change: the transformer forecaster (1L, d=128
# bottleneck) is replaced by K=12 linear CPC heads (van den Oord et al. 2018)
# + the `cpc_multistep` multi-step InfoNCE. No bottleneck flags — the linear
# heads ARE the forecaster.
#
# 1-GPU bs256 (== 2-GPU bs128 per train.py's gathered-loss note); we run one
# seed per GPU so two seeds train concurrently on elisa's two 4090s.
# Code from the WORKTREE; checkpoints/logs to the MAIN checkout (CLAUDE.md
# rule 4 — survive `git worktree remove`).
#
# Usage:  elisa_run.sh <seed> <gpu> [prec]
#   seed = e.g. 20260520 (β's seed) | 20260523
#   gpu  = CUDA_VISIBLE_DEVICES index (0|1)
#   prec = fp16 (default, β's precision) | fp32
set -uo pipefail
SEED="${1:?seed}"; GPU="${2:?gpu}"; PREC="${3:-fp16}"
case "$PREC" in
  fp16) DT=(--residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32) ;;
  fp32) DT=(--residual-dtype fp32 --attn-dtype fp32 --ffn-dtype fp32 --conv-dtype fp32 --patch-emb-dtype fp32) ;;
  *) echo "unknown prec $PREC (fp16|fp32)"; exit 2 ;;
esac
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
NAME="bb_cpc_k12_s${SEED}_${PREC}_50k"
TOTAL=50000
RUNS="$MAIN/runs"; RES="$MAIN/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [cpc s$SEED g$GPU] $*"; }

[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
export CUDA_VISIBLE_DEVICES="$GPU"
log "BB START K=12 linear CPC β2=0.98 τ=0.10 prec=$PREC 1-GPU bs256 GPU=$GPU -> $TOTAL"
python3 -u "$TRAIN" \
  --batch-size 256 --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 \
  --forecaster-kind linear_cpc --cpc-k-steps 12 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cpc_multistep --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  "${DT[@]}" >>"$tlog" 2>&1
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
