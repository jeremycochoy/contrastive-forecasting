#!/bin/bash
# arm 5 (allt·0.8%, mix 0.0078125) on the vast on-demand RTX 6000 Ada — replicates
# train_backbone_b1024_1gpu.sh's exact command (qk+aon, LR1e-3, 12.5k steps).
set -uo pipefail
cd /root/cf
export PYTHONPATH=/root/cf
export HF_TOKEN="$(cat /root/cf/experiments/hf_token.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4 XSHH_ALLT_CHUNK=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
mkdir -p /root/runs
NAME=bb_xshh_allt_forked2_qk_aon_6Lf_b1024
# resume if a periodic checkpoint already exists (idempotent / restart-safe)
RESUME=""; latest=$(ls -t /root/runs/${NAME}_*k.pth 2>/dev/null | head -1)
[ -n "$latest" ] && RESUME="--resume $latest"
nohup python -u experiments/2026-04-27_freq-embedding/scripts/train.py $RESUME \
  --qk-norm --attn-out-norm \
  --batch-size 1024 --device cuda --total-steps 12500 --lr 1e-3 --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 --seed 20260520 \
  --save-every 2500 --save-dir /root/runs --run-name "$NAME" --log-every 100 \
  --hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 \
  --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6 \
  --num-encoder-layers 6 --num-layers 6 \
  --forecaster-d-model 128 --forecaster-n-heads 4 \
  --encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers \
  --depthwise-conv 3 --deprecated-depthwise-conv 0 \
  --loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor \
  --tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --encoder-type gru \
  --synth-kind forked-arma --mix-ratio 0.0078125 \
  --mixup-p 0.3 --freq-emb-dim 3 --seasonality-emb-dim 3 \
  --log-attn-amplitude --log-attn-amplitude-every 200 \
  --residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16 --conv-dtype fp16 --patch-emb-dtype fp32 \
  > /root/run_arm5.log 2>&1 &
echo "arm5-pid $!"
