#!/bin/bash
# R8_E10: 12L causal transformer with Gaussian NLL output.
#
# Round 5/7 plateaued at ~0.192 ema across every quantile variant
# regardless of size or training length — strong evidence the pinball
# loss surface is at its noise floor. Gaussian NLL has smooth gradient
# everywhere; closed-form quantiles via inverse normal CDF at eval.
#
# Same recipe as R5_E7 (the winner @ 1.002 triage GM-MASE) but with
# `--head-arch transformer-gaussian`. 12L causal, Moirai HP, cosine
# warmup=2000, decay 48k→60k to 0.1*peak, 60k steps total.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E10="R8_E10_xfmr12L_gauss_moirai_cosine_60k"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R8 STAGE E10: $E10 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer-gaussian --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 60000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir checkpoints --run-name "$E10" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E10}_best.pth" "checkpoints/${E10}_FINAL.pth"
echo "=== R8 STAGE E10 DONE ===" && date
echo "" && echo "=== R8 ALL DONE ===" && date
