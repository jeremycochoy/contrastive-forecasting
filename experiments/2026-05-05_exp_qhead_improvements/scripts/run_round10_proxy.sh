#!/bin/bash
# R10: Proxy test — train an R3_E4-recipe head on each of 5 backbones,
# triage them downstream. Goal: see which backbone metric (R², AUC, U)
# correlates with downstream GIFT-Eval MASE.
#
# Recipe (matches R3_E4, the first transformer breakthrough):
#   - 6L causal transformer head, H=384 nhead=6, dropout=0.1
#   - Moirai HP: lr=1e-3, β1=0.9, β2=0.98, wd=0.1
#   - cosine schedule warmup=1000, decay 24k→30k, final 0.1×peak
#   - 30k steps, bs=256, forecast_len=16
#   - --head-train-input f_only (legacy default — keeps it simple/cheap)
#
# Five backbones (must match the cross-backbone metric eval set):
#   moirai_hp_early       (sync_realonly_full4096_moirai_hp/.../best_loss)
#   FRESH_50k             (sync_realonly_full4096_moirai_hp_FRESH/.../best_loss)
#   backbone_beta_167k    (current best — sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/.../best_loss)
#   moirai_hp_FINAL_run1  (sync_realonly_full4096_moirai_hp_FINAL_run1/.../180k)
#   learnable_tau         (sync_realonly_full4096_learnable_tau/.../best_loss)

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"

train_one () {
    local NAME="$1"
    local BB_PATH="$2"
    local RUN_NAME="R10_proxy_${NAME}"

    if [ ! -f "$BB_PATH" ]; then
        echo "[R10] SKIP $NAME — missing checkpoint at $BB_PATH" >&2
        return 0
    fi
    if [ -f "checkpoints/${RUN_NAME}_FINAL.pth" ]; then
        echo "[R10] SKIP $NAME — already trained (checkpoints/${RUN_NAME}_FINAL.pth exists)"
        return 0
    fi

    echo "" && echo "=== R10 BACKBONE: $NAME ($(basename $BB_PATH)) ===" && date
    python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB_PATH" --forecast-len 16 \
        --quantile-head --head-arch transformer --head-causal true \
        --head-num-layers 6 --head-nhead 6 --head-ffn-mult 4.0 \
        --head-dropout 0.1 \
        --total-steps 30000 --batch-size 256 \
        --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
        --schedule cosine --warmup-steps 1000 --final-lr-ratio 0.1 \
        --save-every 5000 --log-every 500 \
        --save-dir checkpoints --run-name "$RUN_NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --mix-ratio 0.0 \
        --rev-norm-kind ewma --rev-norm-span 128 \
        --reconstruction forecaster
    cp -f "checkpoints/${RUN_NAME}_best.pth" "checkpoints/${RUN_NAME}_FINAL.pth"
    echo "=== R10 BACKBONE $NAME DONE ===" && date
}

# Five backbones in increasing order of expected wall-clock cost.
# (All are 6L H=384 — same head training time.)
train_one "moirai_hp_early"      "checkpoints/backbone_moirai_hp_early.pth"
train_one "FRESH_50k"            "checkpoints/backbone_FRESH_50k.pth"
train_one "backbone_beta_167k"   "checkpoints/backbone_beta_167k.pth"
train_one "moirai_hp_FINAL_run1" "checkpoints/backbone_moirai_hp_FINAL_run1.pth"
train_one "learnable_tau"        "checkpoints/backbone_learnable_tau.pth"

echo "" && echo "=== R10 PROXY: ALL 5 BACKBONES DONE ===" && date
