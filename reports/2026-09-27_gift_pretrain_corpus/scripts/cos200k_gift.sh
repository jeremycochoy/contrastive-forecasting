#!/bin/bash
# #419: #414's cos200k arm (k3_r100_09_lr56_fix09_dec10k_cos200k, the cell
# arm6_v2_combab_alignT at d_model 384, batch 64, 5.6e-5 cosine to 1e-6 by
# 200k then flat) on the GiftEvalPretrain stream.
#
# The argument list is the cell's first leg as its log records it
# (/workspace/results/run_cf393_arm6_v2_combab_alignT_cf373k3_cf412_k3_r100_
# 09_lr56_fix09_dec10k_cos200k.log, the "Command line:" of the leg to 665k),
# with four changes: --hf-repo/--hf-path become --gift-pretrain --freq-vocab
# v2, and the save dir and the run name are new. The repeated flags of that
# line stay as they were; argparse keeps the last value of each.
#
# One leg to 665,000 steps. A checkpoint every 20,000 steps and one at
# 665,000 give the cos200k stops: 40k 100k 200k 300k 400k 500k 600k 665k.
# CF419_DRY_RUN=1 prints the command and runs nothing.
set -uo pipefail
WT="${WT:-/workspace/cf-419c}"
SAVE="${SAVE:-/workspace/ckpt/cf-419c/cos200k/leg_665k}"
NAME="${NAME:-cf419_cos200k}"
ARGS=(
  --qk-norm --attn-out-norm --batch-size 64 --device cuda --total-steps 665000
  --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
  --seed 20260520 --save-every 20000 --extra-save-steps 665000
  --save-dir "$SAVE" --run-name "$NAME" --log-every 200
  --gift-pretrain --freq-vocab v2
  --t-raw 4096 --n-channels 1 --d-model 64 --n-heads 8
  --num-encoder-layers 3 --num-layers 3 --encoder-dropkey 0.70
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers
  --depthwise-conv 3 --deprecated-depthwise-conv 0
  --loss-shape cosine_similarity_batch_rep_only --align-loss-weight 1.0
  --moco-rep-keys --tau-rep 1.0 --align-target teacher --ema-embedding
  --ema-encoder --ema-tau 0.9 --cpc-infonce-weight 1.0 --sigreg-embedding
  --sigreg-encoding --sigreg-n-chunk 2048 --sigreg-embedding-weight 1.0
  --sigreg-encoding-weight 1.0 --tau 0.10 --rev-norm-kind ewma
  --rev-norm-span 128 --encoder-type gru --synth-kind forked-arma
  --mix-ratio 0.0078125 --crossfade-triplets 1 --mixup-p 0.3
  --freq-emb-dim 3 --seasonality-emb-dim 3 --log-attn-amplitude
  --log-attn-amplitude-every 200 --residual-dtype fp32 --attn-dtype fp16
  --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32
  --train-rollout-depth 3 --cpc-infonce-weight 0.0 --d-model 384
  --n-heads 8 --num-layers 3 --num-encoder-layers 3
  --train-rollout-reduce sum --batch-size 64 --lr 5.6e-5
  --rep-loss-weight 1.0 --rep-loss-weight-end 0.0
  --rep-loss-weight-ramp-steps 10000 --lr-final 1e-6
  --lr-cosine-steps 200000
)
# The environment of #414's leg runner (run_arm_lalign_k.sh).
ENVS=(PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      OMP_NUM_THREADS=8 FCST_GRAD_CKPT=1 XSHH_ALLT_CHUNK=1 CPC_CB_CHUNK=64
      PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 TEACHER_EMBED_CHUNK=16
      CUDA_VISIBLE_DEVICES="${BB_GPU:-0}")
TRAIN="$WT/experiments/2026-04-27_freq-embedding/scripts/train.py"
if [ -n "${CF419_DRY_RUN:-}" ]; then
  echo "${ENVS[*]} HF_TOKEN=\$(cat $WT/experiments/hf_token.txt) python3 -u $TRAIN ${ARGS[*]}"
  exit 0
fi
mkdir -p "$SAVE" || exit 2
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")" HUGGING_FACE_HUB_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
exec env "${ENVS[@]}" python3 -u "$TRAIN" "${ARGS[@]}"
