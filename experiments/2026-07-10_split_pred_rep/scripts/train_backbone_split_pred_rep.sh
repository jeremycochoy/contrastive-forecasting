#!/bin/bash
# #374 — Split the champion loss into L_pred + L_rep.
#
# Same recipe as the champion arm C recipe from #366
# (experiments/2026-06-28_sigreg_lambda_tau_cross/scripts/train_backbone_sigreg.sh
# invoked with λ_e=1, λ_h=1, τ=0.90 by launch_arms_cd.sh) except:
#   --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt
#     → cosine_similarity_batch_split_pred_rep
#   --pos-in-denominator          — DROPPED (L_pred is normalized by construction)
#   --subtract-contrastive-floor  — DROPPED (formula derived for the combined shape)
# CPC / SIGReg (λ_e=λ_h=1.0) / EMA-teacher / τ=0.10 / B=512 / seed / 12,500
# steps unchanged; head protocol is teacher-forced probe (2L + 6L,
# best-loss + last), full-97 GIFT-Eval B4.
#
#   train_backbone_split_pred_rep.sh <gpu> [steps] [save_every]
set -uo pipefail
GPU="${1:?gpu}"; STEPS="${2:-12500}"; SAVE_EVERY="${3:-2500}"
SEED=20260520
WT="${WT:?WT (worktree root) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
OUT="${OUT:-$WT/experiments/2026-07-10_split_pred_rep}"
ENC_LAYERS=6; NENC=3
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
NAME="bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
export TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-split-pred-rep g$GPU] $*"; }
[ -n "$HF_TOKEN" ] || { log "ABORT: empty HF_TOKEN — HF stream would throttle GPU"; exit 1; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START arm=split-pred-rep bs=512 steps=$STEPS xshh_chunk=$XSHH_ALLT_CHUNK cpc_chunk=$CPC_CB_CHUNK ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_split_pred_rep \
  --ema-embedding --ema-encoder --ema-tau 0.9 --cpc-infonce-weight 1.0 \
  --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
  --sigreg-embedding-weight 1.0 --sigreg-encoding-weight 1.0 \
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
