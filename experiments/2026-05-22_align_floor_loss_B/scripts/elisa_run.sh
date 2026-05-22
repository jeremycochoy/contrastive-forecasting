#!/bin/bash
# #313 — (B) recipe + L_align (BYOL) + InfoNCE-floor subtraction.
#
# (B) baseline = experiments/2026-05-19_crossed_loss_ablation cl_hh_50k:
#   bottleneck forecaster d=128/h=4, fp16 body (fp32 residual+patch-emb),
#   τ=0.10, β2=0.95, dropkey 0.70 shared, loss hh_negs, pos-in-denom,
#   seed 20260520, 50k steps.  full-97 GM-MASE 1.3572.
#
# Two opt-in loss flags added on top (PR #312), EVERYTHING else identical:
#   --align-loss-weight 1.0       L_align = λ·(2 − 2·cos(f_t, sg(h_{t+1})))
#                                 added to the loss → AFFECTS gradients.
#   --subtract-contrastive-floor  re-base loss by log(1+N·e^(−1/τ)).
#                                 GRADIENT-NEUTRAL (a constant) → only makes
#                                 the logged loss read ~0 at convergence.
# ⇒ the only training change vs (B) is L_align; the floor is cosmetic.
#
# Code from the WORKTREE; outputs to the MAIN checkout (CLAUDE.md).
# Auto-DDP when ≥2 GPUs visible (global batch 256), else 1-GPU bs256
# (identical loss per train.py:739-740).
#
# Usage:  elisa_run.sh [gpus]
#   gpus = CUDA_VISIBLE_DEVICES (default "0,1"); pass "1" to grab one GPU.
set -uo pipefail
GPUS="${1:-0,1}"
SEED=20260520
WT=/home/jupyter/cf-wt-align-floor
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B
NAME="bb_alignfloor_50k"
TOTAL=50000
RUNS="$MAIN/runs"; RES="$MAIN/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null || cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [alignfloor] $*"; }
freeport(){ python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()'; }

[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
export CUDA_VISIBLE_DEVICES="$GPUS"
ng=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null)
if [ "${ng:-0}" -ge 2 ]; then
  LAUNCHER=(torchrun --nproc_per_node=2 --master_port="$(freeport)"); PER_RANK_BS=128
  log "BB START align+floor DDP nproc=2 bs128 GPUs=$GPUS -> $TOTAL"
else
  LAUNCHER=(python3 -u); PER_RANK_BS=256
  log "BB START align+floor 1-GPU bs256 GPUs=$GPUS -> $TOTAL"
fi
"${LAUNCHER[@]}" "$TRAIN" \
  --batch-size "$PER_RANK_BS" --device cuda --total-steps "$TOTAL" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.95 --seed "$SEED" \
  --save-every 5000 --save-dir "$RUNS" --run-name "$NAME" \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 1 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs --pos-in-denominator \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --align-loss-weight 1.0 --subtract-contrastive-floor \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
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
