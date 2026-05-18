#!/bin/bash
# τ-sweep proxy MASE: train an R3_E4-recipe head on each of 5 τ-sweep
# backbones (FINAL.pth), then triage-eval to obtain per-arm proxy GM-MASE.
# Mirrors run_round10_proxy.sh from 2026-05-05_exp_qhead_improvements.
#
# Recipe (R3_E4, the first transformer breakthrough):
#   - 6L causal transformer head, H=384 nhead=6, dropout=0.1
#   - Moirai HP: lr=1e-3, β1=0.9, β2=0.98, wd=0.1
#   - cosine schedule warmup=1000, decay 24k→30k, final 0.1×peak
#   - 30k steps, bs=256, forecast_len=16
#   - --head-train-input f_only (legacy default — keeps it simple/cheap)

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"

train_one () {
    local TAU="$1"
    local SAFE_TAU=${TAU//./_}
    local NAME="tau_sweep_${SAFE_TAU}"
    local BB_PATH="checkpoints/${NAME}_FINAL.pth"
    local RUN_NAME="R10_proxy_${NAME}"

    if [ ! -f "$BB_PATH" ]; then
        echo "[proxy] SKIP $NAME — missing checkpoint at $BB_PATH" >&2
        return 0
    fi
    if [ -f "checkpoints/${RUN_NAME}_FINAL.pth" ]; then
        echo "[proxy] SKIP $NAME — already trained (checkpoints/${RUN_NAME}_FINAL.pth exists)"
        return 0
    fi

    echo "" && echo "=== PROXY HEAD τ=${TAU} → backbone ${NAME} ===" && date
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
    echo "=== PROXY HEAD τ=${TAU} DONE ===" && date
}

TAUS=(0.03 0.05 0.07 0.10 0.20)
for TAU in "${TAUS[@]}"; do
    train_one "${TAU}"
done

echo "" && echo "=== τ sweep proxy: ALL 5 HEADS DONE ===" && date
