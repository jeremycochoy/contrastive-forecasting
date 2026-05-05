#!/bin/bash
# Round 1 — qhead improvements on backbone-beta.
#
# Two experiments back-to-back on the same vast.ai 5090:
#   E1: linear-probe quantile head + legacy lr=3e-4  (diagnostic)
#   E2: GRU quantile head + Moirai HP + WSD schedule (best-bet improvement)
#
# Backbone-beta = checkpoints/tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth
# (synced from the #10 RESUME50k run; GM-MASE 1.183 with the legacy qhead).
#
# Eval runs separately on elisa (free 4090s, GIFT-Eval data already on disk).

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"

E1="R1_E1_linear_quant_lr3e4"
E2="R1_E2_gru_quant_moirai_wsd"

if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "ERROR: backbone-beta not found at checkpoints/${BB}_FINAL.pth" >&2
    exit 1
fi

# --- E1: Linear probe -------------------------------------------------------
# Diagnostic: single-Linear quantile head on top of frozen backbone.
# If this scores close to E2 (or close to the legacy 1.183 GRU-head baseline),
# head capacity isn't the bottleneck and effort goes to schedule/HP.
echo "" && echo "=== R1 STAGE E1: $E1 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch linear \
    --total-steps 30000 --batch-size 256 --lr 3e-4 \
    --save-every 1000 --log-every 200 \
    --save-dir checkpoints --run-name "$E1" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E1}_best.pth" "checkpoints/${E1}_FINAL.pth"
echo "=== R1 STAGE E1 DONE ===" && date

# --- E2: GRU + Moirai HP + WSD ---------------------------------------------
# Moirai HP for AdamW: β2=0.98, wd=0.1 (matches the backbone training recipe).
# WSD schedule: linear warmup 0→peak over 500 steps, stable peak from 500..24000
# (80% of total), linear cooldown 24000..30000 to 0.1*peak.
# The 24k checkpoint is the "stable" branchable point (MiniCPM/Hägele).
# Periodic saves every 1k step → ckpts at 1k,...,24k,...,30k for fan-out.
echo "" && echo "=== R1 STAGE E2: $E2 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 \
    --quantile-head --head-arch gru \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule wsd --warmup-steps 500 \
    --decay-start-step 24000 --final-lr-ratio 0.1 \
    --save-every 1000 --log-every 200 \
    --save-dir checkpoints --run-name "$E2" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${E2}_best.pth" "checkpoints/${E2}_FINAL.pth"
# Also tag the stable-end checkpoint for later cooldown branching.
if [ -f "checkpoints/${E2}_24k.pth" ]; then
    cp -f "checkpoints/${E2}_24k.pth" "checkpoints/${E2}_STABLE.pth"
    cp -f "checkpoints/${E2}_24k_optimizer.pth" \
          "checkpoints/${E2}_STABLE_optimizer.pth"
fi
echo "=== R1 STAGE E2 DONE ===" && date

echo "" && echo "=== R1 ALL DONE ===" && date
