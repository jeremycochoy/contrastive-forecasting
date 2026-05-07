#!/bin/bash
# R9_E14: 100k version of R9_E13 (the new winner @ 0.990 triage GM-MASE).
# Tests if longer training under the matched-input recipe helps.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E14="R9_E14_xfmr12L_quant_moirai_cosine_e_then_f_100k"

echo "" && echo "=== R9 STAGE E14: $E14 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --head-train-input e_then_f \
    --total-steps 100000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 3000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir checkpoints --run-name "$E14" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E14}_best.pth" "checkpoints/${E14}_FINAL.pth"
echo "=== R9 STAGE E14 DONE ===" && date
echo "" && echo "=== R9 ALL DONE ===" && date
