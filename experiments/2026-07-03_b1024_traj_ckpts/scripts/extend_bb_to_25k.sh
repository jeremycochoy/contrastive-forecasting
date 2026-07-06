#!/bin/bash
# #369 follow-up — resume the arm-C B=1024 backbone from step 12500 and
# train to step 25000 with fine trajectory saves every 500 steps.
# Emits `_r1_step<N>.pth` files (train.py auto-suffixes run-name to
# protect the original _step500…_step12500 set).
#
# Usage:
#   WT=/home/jupyter/cf-369 OUT=/home/jupyter/cf-369/experiments/2026-07-03_b1024_traj_ckpts \
#     GPU=1 bash extend_bb_to_25k.sh
set -uo pipefail
: "${WT:?}"; : "${OUT:?}"
GPU="${GPU:-1}"
STEPS="${STEPS:-25000}"
TRAJ_SAVE_EVERY="${TRAJ_SAVE_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
SEED=20260520
ENC_LAYERS=6; NENC=3

. "$OUT/winners.sh"
LAMBDA_E="${LAMBDA_E:?}"; LAMBDA_H="${LAMBDA_H:?}"; TAU="${TAU:?}"

RUNS="$OUT/runs"; RES="$OUT/results"
# Derive NAME from the actual _12k.pth on disk to avoid float-format drift.
RESUME_FROM="$(ls -t "$RUNS/"bb_*_b1024_12k.pth 2>/dev/null | grep -v optimizer | head -1)"
[ -n "$RESUME_FROM" ] && [ -f "$RESUME_FROM" ] \
  || { echo "ABORT: could not locate a bb_..._b1024_12k.pth in $RUNS" >&2; exit 2; }
NAME="$(basename "$RESUME_FROM" _12k.pth)"
SUFFIX="${NAME#bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_}"

HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
[ -f "$HF_TOKEN_PATH" ] || { echo "ABORT: HF token missing" >&2; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
[ -f "$TRAIN" ] || { echo "ABORT: TRAIN=$TRAIN not found" >&2; exit 2; }

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-8}"
export CUDA_VISIBLE_DEVICES="$GPU"

tlog="$RES/run_bb_extend25k_${SUFFIX}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-ext25k-${SUFFIX} g$GPU] $*"; }
log "BB EXTEND START steps=$STEPS traj_save_every=$TRAJ_SAVE_EVERY resume=$(basename "$RESUME_FROM")"

python3 -u "$TRAIN" --resume "$RESUME_FROM" --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --traj-save-every "$TRAJ_SAVE_EVERY" \
  --save-dir "$RUNS" --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers "$NENC" --num-layers "$ENC_LAYERS" \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --ema-embedding --ema-encoder --ema-tau "$TAU" --cpc-infonce-weight 1.0 \
  --sigreg-embedding --sigreg-encoding --sigreg-n-chunk 2048 \
  --sigreg-embedding-weight "$LAMBDA_E" --sigreg-encoding-weight "$LAMBDA_H" \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 --crossfade-triplets 1 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  >>"$tlog" 2>&1
rc=$?
log "BB EXTEND rc=$rc"
exit $rc
