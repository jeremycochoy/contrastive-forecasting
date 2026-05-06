#!/bin/bash
# R5_E7: combine the depth + length wins from Round 4.
#
# R4_E5 (6L × 60k cosine) → 1.009 GM-MASE (length axis: −0.8% over R3_E4)
# R4_E6 (12L × 30k cosine) → 1.005 GM-MASE (depth axis: −1.2% over R3_E4)
#
# R5_E7 stacks both: 12L × 60k cosine. If gains stack linearly, we'd
# expect ~0.99. Diminishing returns would land at ~1.00. Worse than
# either of the components would mean training instability or capacity
# saturation at this head structure.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E7="R5_E7_xfmr12L_quant_moirai_cosine_60k"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R5 STAGE E7: $E7 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 60000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir checkpoints --run-name "$E7" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E7}_best.pth" "checkpoints/${E7}_FINAL.pth"
echo "=== R5 STAGE E7 DONE ===" && date
echo "" && echo "=== R5 ALL DONE ===" && date
