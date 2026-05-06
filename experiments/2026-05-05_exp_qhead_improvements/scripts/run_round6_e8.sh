#!/bin/bash
# R6_E8: bidirectional transformer + forecast_len=128 + B1 eval.
#
# Round 5 confirmed the transformer head plateaus at ~0.192 ema_loss
# regardless of size or training length (R3_E4 6L×30k, R4_E5 6L×60k,
# R4_E6 12L×30k, R5_E7 12L×60k all converged to the same loss floor).
# That's the loss surface, not the model.
#
# Two angles attacked together:
#   - forecast_len=128 instead of W=16: 8× more target values per step
#     → more supervision per gradient. Eval-time B1 strategy already
#     decodes 128 values per rolled position.
#   - --head-causal false: at output position t the head can attend to
#     f_t, f_{t+1}, …, f_{t+k} — necessary for reconstructing the
#     128-value target span, which extends across patches t+1..t+8.
#
# 6L × 30k matches R3_E4's training budget so the comparison is clean.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
E8="R6_E8_xfmr6L_bidir_fl128_quant_moirai_cosine"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

echo "" && echo "=== R6 STAGE E8: $E8 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 128 \
    --quantile-head --head-arch transformer --head-causal false \
    --head-num-layers 6 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
    --save-every 2000 --log-every 200 \
    --save-dir checkpoints --run-name "$E8" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E8}_best.pth" "checkpoints/${E8}_FINAL.pth"
echo "=== R6 STAGE E8 DONE ===" && date
echo "" && echo "=== R6 ALL DONE ===" && date
