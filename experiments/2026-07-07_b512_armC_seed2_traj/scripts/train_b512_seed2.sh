#!/bin/bash
# #371 — B=512 arm-C seed-control retrain with trajectory checkpoints.
# Same flags as #366's train_backbone_sigreg.sh arm C, except:
# seed 20260707 (control), 25,000 steps, --traj-save-every 500, and the
# save dir lives in the MAIN CHECKOUT (storage directive), not a worktree.
#
#   WT=/home/jupyter/cf-371 EXP=<this experiment dir> GPU=0 bash train_b512_seed2.sh
set -uo pipefail
: "${WT:?}"; : "${EXP:?}"
GPU="${GPU:-0}"
STEPS="${STEPS:-25000}"
TRAJ_SAVE_EVERY="${TRAJ_SAVE_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
SEED=20260707            # control seed — original arm C used 20260520
LAMBDA_E=1.0; LAMBDA_H=1.0; TAU=0.90
ENC_LAYERS=6; NENC=3

SAVE_DIR="${SAVE_DIR:-/home/jupyter/contrastive-forecasting/sync_b512_armC_seed2}"
case "$SAVE_DIR" in
  */cf-3*|*/worktrees/*) echo "ABORT: save dir $SAVE_DIR is inside a worktree" >&2; exit 2 ;;
esac
mkdir -p "$SAVE_DIR"
RES="$EXP/results"; mkdir -p "$RES"

[ $((STEPS % TRAJ_SAVE_EVERY)) -eq 0 ] \
  || { echo "ABORT: STEPS=$STEPS not a multiple of TRAJ_SAVE_EVERY=$TRAJ_SAVE_EVERY" >&2; exit 2; }

NAME="bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lC_emb10_enc10_tau090_seed2"

HF_TOKEN_PATH="$WT/experiments/hf_token.txt"
[ -f "$HF_TOKEN_PATH" ] || { echo "ABORT: HF token missing at $HF_TOKEN_PATH" >&2; exit 2; }
export HF_TOKEN="$(cat "$HF_TOKEN_PATH")"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "ABORT: empty HF_TOKEN" >&2; exit 2; }

TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
[ -f "$TRAIN" ] || { echo "ABORT: TRAIN=$TRAIN not found" >&2; exit 2; }
grep -q -- "--traj-save-every" "$TRAIN" \
  || { echo "ABORT: $TRAIN lacks --traj-save-every (branch missing #369's train.py commits)" >&2; exit 2; }

export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK="${CPC_CB_CHUNK:-64}"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 TEACHER_EMBED_CHUNK="${TEACHER_EMBED_CHUNK:-16}"
export CUDA_VISIBLE_DEVICES="$GPU"

tlog="$RES/run_${NAME}.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [bb-b512-seed2 g$GPU] $*"; }
log "BB START seed=$SEED bs=512 steps=$STEPS traj_save_every=$TRAJ_SAVE_EVERY save_dir=$SAVE_DIR"

python3 -u "$TRAIN" --qk-norm --attn-out-norm \
  --batch-size 512 --device cuda --total-steps "$STEPS" --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed "$SEED" \
  --save-every "$SAVE_EVERY" --traj-save-every "$TRAJ_SAVE_EVERY" \
  --save-dir "$SAVE_DIR" --run-name "$NAME" --log-every 100 \
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
log "BB rc=$rc"
exit $rc
