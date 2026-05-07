#!/bin/bash
# R9_E11: train head on [e_flat, f_flat] sequence (length 2T). Fixes
# train-eval input distribution mismatch — at training the head used to
# see f_0..f_{T-1} only, but at eval the B-strategies feed [e_ctx,
# rolled_f]. R9_E11 uses --head-train-input e_then_f which concatenates
# encoder + forecaster latents at training so the head sees the same
# input layout it'll see at eval.
#
# Same recipe as R5_E7 (the winner @ 1.002 triage):
#   12L causal transformer, Moirai HP β2=0.98 wd=0.1 lr=1e-3,
#   cosine warmup=2000 → 0.1×peak, 60k steps, bs=256, forecast_len=16.
#
# Memory budget: 2T = 512 tokens at training (vs 256 in R5_E7). With
# Flash-attention (PyTorch SDPA default) memory is linear in T so this
# should fit on a 4090 at bs=256. Will reduce bs if OOM.
#
# Runs on elisa GPU 0 (vast credit exhausted).

set -e
ROOT="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH=.
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=0

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB_PATH="${ROOT}/sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/moirai_hp_FRESH_RESUME50k/checkpoints/tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth"
SAVE_DIR="${ROOT}/sync_qhead_beta_rd1/checkpoints"
E11="R9_E11_xfmr12L_quant_moirai_cosine_e_then_f_60k"

mkdir -p "$SAVE_DIR"

echo "" && echo "=== R9 STAGE E11: $E11 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --head-train-input e_then_f \
    --total-steps 60000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$E11" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "${SAVE_DIR}/${E11}_best.pth" "${SAVE_DIR}/${E11}_FINAL.pth"
echo "=== R9 STAGE E11 DONE ===" && date
