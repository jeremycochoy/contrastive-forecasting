#!/bin/bash
# #419 throughput smoke: the #415 Moirai-schedule leg, flag for flag, with
# the old data path (small_v1) or the new one (--gift-pretrain). Only the
# data flags, the step count, the save dir and the log cadence differ.
# Usage: smoke.sh old|new <steps>. WT is the checkout, OUT the scratch root.
mode="${1:?old|new}"; steps="${2:?steps}"
WT="${WT:-/workspace/cf-419}"; OUT="${OUT:-/workspace/cf419-scratch}"
cd "$WT" || exit 2
export HF_TOKEN="$(cat experiments/hf_token.txt)" HUGGING_FACE_HUB_TOKEN="$(cat experiments/hf_token.txt)"
export PYTHONPATH="$WT" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 FCST_GRAD_CKPT=1 PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4
if [ "$mode" = old ]; then
  DATA=(--hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1)
else
  DATA=(--gift-pretrain --freq-vocab v2)
fi
out="$OUT/smoke_$mode"
mkdir -p "$out"
CUDA_VISIBLE_DEVICES=0 exec python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
  --qk-norm --attn-out-norm --batch-size 256 --device cuda --total-steps "$steps" \
  --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
  --lr-final 0 --lr-cosine-steps 166000 --lr-warmup-steps 10000 --grad-clip 1.0 \
  --seed 20260520 --save-every 1000000 --save-dir "$out" --run-name "smoke_$mode" \
  --log-every 25 "${DATA[@]}" --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 8 \
  --num-encoder-layers 3 --num-layers 3 --encoder-dropkey 0.70 \
  --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 --rev-norm-kind ewma \
  --rev-norm-span 128 --encoder-type gru --synth-kind forked-arma \
  --mix-ratio 0.0078125 --crossfade-triplets 1 --mixup-p 0.3 --freq-emb-dim 3 \
  --seasonality-emb-dim 3 --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 \
  --patch-emb-dtype fp32 --value-space-objective --train-rollout-depth 3 \
  --train-rollout-reduce sum > "$out/run.log" 2>&1
