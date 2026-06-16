#!/bin/bash
# #344 follow-up arm — enc6 + CPC InfoNCE + BYOL align, NO main contrastive loss.
# Tests whether CPC (single-step InfoNCE via the learnable bilinear W_1) plus a
# separate forecaster loss (BYOL align, encoder target stop-gradded) beats the
# elaborate xshh_allt contrastive objective. Everything else is the exact enc6
# baseline recipe.
#
# Runs 2-GPU DDP (torchrun): --batch-size 512 PER RANK ⇒ GLOBAL batch 1024
# (== the single-GPU baselines), and the loss is computed on the gathered
# global batch (`gather_latents`, "global negatives, == 1-GPU @ global B") — so
# CPC's cross-batch negatives and align span both GPUs' samples. ~2× faster than
# single-GPU. On crash, --resume the latest periodic checkpoint (full state),
# and train.py appends to the SAME losses CSV (continuous, conserved).
#   train_backbone_cpcalign.sh [steps] [save_every]
set -uo pipefail
STEPS="${1:-12500}"; SAVE_EVERY="${2:-2500}"
SEED=20260520
WT="${WT:-/home/jupyter/contrastive-forecasting/.claude/worktrees/exp+cpc-infonce-344}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
NAME="bb_allt08_xftrip_nobn_enc6_cpcalign_qk_aon_b1024_cpc"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-cpcalign-ddp] $*"; }
[ -n "$HF_TOKEN" ] || log "WARN: empty HF_TOKEN — HF stream will throttle the GPUs"
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
MP=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
log "BB START 2-GPU DDP (per-rank bs=512, global 1024) cpc+align/no-main steps=$STEPS port=$MP ${RESUME}"
torchrun --nproc_per_node=2 --master_port="$MP" "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt \
  --no-main-contrastive-loss --align-loss-weight 1.0 --cpc-infonce-weight 1.0 \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (--resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
