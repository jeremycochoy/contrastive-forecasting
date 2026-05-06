#!/bin/bash
# R4_E6: deeper transformer (12 layers vs R3_E4's 6) on elisa GPU 0.
#
# Tests depth axis independent of length. Same recipe as R3_E4 but
# num_layers=12 (~21M params, vs 10.7M). Stays at 30k steps so we can
# compare directly — depth helped if R4_E6's eval beats R3_E4's 1.017.
#
# Runs on elisa GPU 0 (no vast cost). Concurrent with R4_E5 (6L × 60k
# on vast) — if both win equally, combine 12L+60k in R5; if only R4_E5
# wins, length is more important than depth; etc.

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

E6="R4_E6_xfmr12L_quant_moirai_cosine"

if [ ! -f "$BB_PATH" ]; then
    echo "ERROR: backbone-beta missing at $BB_PATH" >&2
    exit 1
fi
mkdir -p "$SAVE_DIR"

echo "" && echo "=== R4 STAGE E6: $E6 (elisa GPU 0) ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch transformer \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
    --save-every 2000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$E6" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "${SAVE_DIR}/${E6}_best.pth" "${SAVE_DIR}/${E6}_FINAL.pth"
echo "=== R4 STAGE E6 DONE ===" && date
