#!/bin/bash
# R9_E13: same recipe as R9_E11 (12L causal transformer + Moirai HP +
# cosine + --head-train-input e_then_f) but at 60k steps. Tests whether
# the e_then_f training fix benefits from R5_E7's full-length budget.
# Runs on vast.ai 5090 (parallel with R9_E11 on elisa GPU 0 at 30k).

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E13="R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R9 STAGE E13: $E13 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --head-train-input e_then_f \
    --total-steps 60000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir checkpoints --run-name "$E13" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E13}_best.pth" "checkpoints/${E13}_FINAL.pth"
echo "=== R9 STAGE E13 DONE ===" && date
echo "" && echo "=== R9 ALL DONE ===" && date
