#!/bin/bash
# #353 — 2-GPU DDP version of train_backbone_ema.sh. Same recipe, same flags,
# same global batch (per-rank 512 × 2 ranks = global 1024); just torchrun and
# the gathered-loss path (`gather_latents` → identical objective to single-GPU
# @ B=1024). Resume from the same checkpoint name as the single-GPU launcher
# so a partial single-GPU run can switch onto two GPUs at the next periodic
# checkpoint with no objective change.
#
#   train_backbone_ema_ddp.sh [steps] [save_every]
set -uo pipefail
STEPS="${1:-12500}"; SAVE_EVERY="${2:-2500}"
SEED=20260520
WT="${WT:-/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-ema-targets-353}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-19_ema_target_encoder}"
NAME="bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
# Per-rank memory now sees B=512 instead of 1024 — Gram is 4× smaller, but
# the cross-batch logsumexp still chunks at 1 to stay headroom-safe under
# the GRU patch-embed and teacher-path memory.
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-4}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-ema-ddp] $*"; }
[ -n "$HF_TOKEN" ] || log "WARN: empty HF_TOKEN — HF stream will throttle the GPUs"
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
MP=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
log "BB START 2-GPU DDP (per-rank bs=512, global 1024) ema-target steps=$STEPS port=$MP ${RESUME}"
torchrun --nproc_per_node=2 --master_port="$MP" "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 3 --num-layers 6 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --ema-embedding --ema-encoder --ema-tau 0.99 --cpc-infonce-weight 1.0 \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL. tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
