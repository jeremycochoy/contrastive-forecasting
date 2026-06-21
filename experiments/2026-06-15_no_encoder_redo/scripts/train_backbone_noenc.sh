#!/bin/bash
# #348 — redo #344's two competing setups WITHOUT the encoder stack
# (only the GRU patch-embedding + the forecaster; --num-encoder-layers 0).
# Two arms, identical to their #344 counterparts except the encoder is removed:
#   base  normal contrastive loss            (no --cpc-infonce-weight)
#   cpc   contrastive loss + CPC InfoNCE     (--cpc-infonce-weight 1.0)
# Everything else is the EXACT #339/#341/#344 recipe (allt·0.8% + crossfade
# triplet, qk-norm, attn-out-norm, xshh_allt loss, --pos-in-denominator,
# --subtract-contrastive-floor, --stopgrad-positive-h, tau 0.10, batch 1024,
# 12 500 steps, seed 20260520, d_model 384 / 6 heads, 6-layer forecaster).
# With num_encoder_layers=0 the forecaster's encoder ModuleList is empty and
# the contrastive target e_t degenerates to the patch-embedding output.
#
# Memory knobs default ON (same as the enc6 #344 arm) so the run always fits a
# 24 GB card; all are byte-identical to the loss (memory <-> kernel launches).
# Override to 0 for the smoke test to measure native speed.
#
#   train_backbone_noenc.sh <arm: base|cpc> <gpu> [steps] [save_every]
set -uo pipefail
ARM="${1:?arm (base|cpc)}"; GPU="${2:?gpu}"; STEPS="${3:-12500}"; SAVE_EVERY="${4:-2500}"
SEED=20260520
WT="${WT:-/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-no-encoder-348}"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-15_no_encoder_redo}"
case "$ARM" in
  base)   CPC_FLAG=() ;;
  cpc)    CPC_FLAG=(--cpc-infonce-weight 1.0) ;;
  # CPC_All (#348): paper-exact CPC InfoNCE (van den Oord Eq. 4) with the STRICT
  # marginal candidate set — {positive} ∪ every OTHER sequence at all steps
  # (context-independent negatives ⇒ Theorem 1 / MI bound holds exactly). The
  # cross-sequence-all-time Gram is chunked over the source batch (CPC_ALL_CHUNK).
  cpcall) CPC_FLAG=(--cpc-infonce-weight 1.0 --cpc-infonce-negs cross); export CPC_ALL_CHUNK="${CPC_ALL_CHUNK:-32}" ;;
  *) echo "unknown arm: $ARM (want base|cpc|cpcall)"; exit 2 ;;
esac
NAME="bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_${ARM}"
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RUNS" "$RES"
BB="$RUNS/${NAME}_FINAL.pth"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$GPU"
# Memory knobs (default ON; set to 0/empty in env to disable for smoke tests).
export FCST_GRAD_CKPT="${FCST_GRAD_CKPT:-1}" XSHH_ALLT_CHUNK="${XSHH_ALLT_CHUNK:-1}"
export CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT="${PATCH_ENC_CKPT:-1}" PATCH_ENC_CHUNK="${PATCH_ENC_CHUNK:-4}"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-noenc-$ARM g$GPU] $*"; }
[ -n "$HF_TOKEN" ] || { log "WARN: empty HF_TOKEN — HF stream will throttle the GPU"; }
[ -f "$BB" ] && { log "BB SKIP ($NAME FINAL exists)"; exit 0; }
tlog="$RES/run_${NAME}.log"
# Crash resilience: resume from the latest periodic checkpoint (model+opt+step+RNG+data).
RESUME=""; latest=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1)
[ -n "$latest" ] && { RESUME="--resume $latest"; log "RESUME from $(basename "$latest")"; }
log "BB START arm=$ARM bs=1024 steps=$STEPS num_encoder_layers=0 grad_ckpt=$FCST_GRAD_CKPT xshh_chunk=$XSHH_ALLT_CHUNK ${RESUME}"
python3 -u "$TRAIN" $RESUME --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 0 --num-layers 6 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --stopgrad-positive-h "${CPC_FLAG[@]}" \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "BB train exited rc=$rc — NOT creating FINAL (incomplete; --resume next launch). tail: $(tail -3 "$tlog"|tr '\n' ' ')"
  exit 1
fi
if   [ -f "$RUNS/${NAME}_best_loss.pth" ]; then cp -f "$RUNS/${NAME}_best_loss.pth" "$BB"
elif [ -f "$RUNS/${NAME}_final.pth" ];     then cp -f "$RUNS/${NAME}_final.pth"     "$BB"
else cp -f "$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null|head -1)" "$BB" 2>/dev/null; fi
[ -f "$BB" ] && { log "BB DONE -> ${NAME}_FINAL.pth ($(du -h "$BB"|cut -f1))"; exit 0; }
log "BB FAILED no checkpoint"; exit 1
