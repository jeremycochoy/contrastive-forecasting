#!/bin/bash
# R2_E3: combine the wins from Round 1.
#   - linear-probe quantile head (E1's win: -5.5% on triage)
#   - Moirai HP β2=0.98 wd=0.1 + WSD schedule (E2's modest gain)
# Hypothesis: stack improvements; expect <1.066 GM-MASE on triage.
#
# Runs on elisa GPU 0 (24GB free) — no vast.ai cost. Eval runs on GPU 1.

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

E3="R2_E3_linear_quant_moirai_wsd"

if [ ! -f "$BB_PATH" ]; then
    echo "ERROR: backbone-beta missing at $BB_PATH" >&2
    exit 1
fi
mkdir -p "$SAVE_DIR"

echo "" && echo "=== R2 STAGE E3: $E3 ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch linear \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule wsd --warmup-steps 500 \
    --decay-start-step 24000 --final-lr-ratio 0.1 \
    --save-every 1000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$E3" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "${SAVE_DIR}/${E3}_best.pth" "${SAVE_DIR}/${E3}_FINAL.pth"
# Also tag the stable-end (24k) checkpoint for later cooldown branching.
if [ -f "${SAVE_DIR}/${E3}_24k.pth" ]; then
    cp -f "${SAVE_DIR}/${E3}_24k.pth" "${SAVE_DIR}/${E3}_STABLE.pth"
    cp -f "${SAVE_DIR}/${E3}_24k_optimizer.pth" "${SAVE_DIR}/${E3}_STABLE_optimizer.pth"
fi
echo "=== R2 STAGE E3 DONE ===" && date
