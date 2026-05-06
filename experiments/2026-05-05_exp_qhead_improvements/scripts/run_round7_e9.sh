#!/bin/bash
# R7_E9: same recipe as R5_E7 (current best, GM-MASE 1.002 triage),
# pushed to 100k steps. R6_E8 (bidir + forecast_len=128) lost ground
# vs R5_E7 due to train-test mismatch (bidir attention on real f's
# at train vs rolled-out f's at eval). Keeping causal + fl=16 and
# just training longer.
#
# 12L × 100k cosine warmup=3000 final=0.1*peak.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E9="R7_E9_xfmr12L_quant_moirai_cosine_100k"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R7 STAGE E9: $E9 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 100000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 3000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir checkpoints --run-name "$E9" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E9}_best.pth" "checkpoints/${E9}_FINAL.pth"
echo "=== R7 STAGE E9 DONE ===" && date
echo "" && echo "=== R7 ALL DONE ===" && date
