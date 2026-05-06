#!/bin/bash
# R4_E5: longer training of R3_E4's winning recipe.
#
# R3_E4 (transformer + Moirai HP + cosine, 30k steps) hit GM-MASE 1.017
# (triage), down from linear's 1.066 plateau. Loss was still dropping at
# step 30000 (final ema_loss=0.192, best=0.192 at step 30000), so longer
# training under cosine should keep pushing.
#
# Schedule scaled to 60k steps:
#   - linear warmup 0..2000 (~3% of total)
#   - cosine decay 2000..60000 from peak lr=1e-3 → 1e-4 (final_ratio=0.1)

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E5="R4_E5_xfmr6L_quant_moirai_cosine_60k"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R4 STAGE E5: $E5 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer \
    --head-num-layers 6 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 60000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir checkpoints --run-name "$E5" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E5}_best.pth" "checkpoints/${E5}_FINAL.pth"
echo "=== R4 STAGE E5 DONE ===" && date
echo "" && echo "=== R4 ALL DONE ===" && date
