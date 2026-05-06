#!/bin/bash
# R3_E4: 6-layer causal-decoder transformer head with Moirai HP + cosine LR.
#
# Hypothesis (per user): a transformer head matching the backbone's depth+
# width gives the head the same representational surface the backbone has,
# trained from scratch with Moirai-style HP and cosine LR. Should jump past
# the linear probe's 1.066 GM-MASE plateau (R1_E1 / R2_E3 both at ~1.067).
#
# Architecture: TransformerQuantileForecastingHead
#   - 6 layers, H=384, nhead=6, ffn_mult=4, dropout=0.1
#   - causal mask (decoder-style)
#   - pre-norm
#   - output: per-step Linear → 9 quantile levels × 16 forecast values
#   - ~10.7M params (between linear's 55k and a 1B-param Moirai)
#
# Schedule (Moirai recipe scaled to 30k steps):
#   - linear warmup 0..1000 steps  (≈3% of total, matching Moirai's ~1%)
#   - cosine decay 1000..30000 from peak lr=1e-3 → 1e-4 (final_ratio=0.1)
#
# AdamW: β1=0.9, β2=0.98, wd=0.1, eps=1e-8 (Moirai HP).
#
# Runs on vast.ai 5090 with cu128 PyTorch (sm_120 support).

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E4="R3_E4_xfmr6L_quant_moirai_cosine"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R3 STAGE E4: $E4 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch transformer \
    --head-num-layers 6 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
    --save-every 2000 --log-every 200 \
    --save-dir checkpoints --run-name "$E4" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E4}_best.pth" "checkpoints/${E4}_FINAL.pth"
echo "=== R3 STAGE E4 DONE ===" && date
echo "" && echo "=== R3 ALL DONE ===" && date
