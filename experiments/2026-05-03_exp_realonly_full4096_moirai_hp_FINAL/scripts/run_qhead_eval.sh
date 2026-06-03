#!/bin/bash
# Continuation of #10 RESUME50k: qhead training + GIFT-eval on the trained
# backbone. Reuses the live vast.ai instance (per CLAUDE.md "edit a
# continuation script in-place and re-launch on the live instance — not
# destroy + re-provision" when a resume bundle is on the box).
#
# Backbone: tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth (= cp of
# best_loss.pth saved at step 166,800 of the 167k-step run).
# Loss-std continuity at the resume boundary verified separately (no jump
# vs the FRESH run's [50k,52.4k] window).

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"

BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
QH="R1q_full4096_moirai_hp_FRESH_RESUME50k"

# Stage the backbone as ${BB}_FINAL.pth so downstream stages find what
# they expect. cp does not delete the source best_loss.pth.
if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    if [ ! -f "checkpoints/${BB}_best_loss.pth" ]; then
        echo "ERROR: best_loss backbone not found at checkpoints/${BB}_best_loss.pth" >&2
        exit 1
    fi
    cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
fi

# RES_DIR matches the existing experiment's path on remote (note: original
# run_fresh.sh used a doubled-date string; preserved for sync_loop
# compatibility). Sync_loop pulls all_results.csv + summary.txt from here.
RES_DIR="experiments/2026-05-03_2026-05-02_exp_realonly_full4096_moirai_hp_FINAL/results/gift_eval_resume50k"
mkdir -p "$RES_DIR"

echo "" && echo "=== run_full4096_moirai_hp_FRESH_RESUME50k: qhead+eval starting ===" && date

echo "" && echo "=== STAGE H: $QH (qhead on RESUME50k backbone, default optim) ===" && date
# Backbone arch must be passed explicitly — head trainer and eval default
# to C=4 / H=512 / nhead=8, our backbone is C=1 / H=384 / nhead=6 (PR #124).
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 256 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QH" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster
cp -f "checkpoints/${QH}_best.pth" "checkpoints/${QH}_FINAL.pth"
echo "=== STAGE H DONE ===" && date

echo "" && echo "=== STAGE E: gift_eval ===" && date
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== STAGE E DONE ===" && date

echo "" && echo "=== run_full4096_moirai_hp_FRESH_RESUME50k: ALL DONE ===" && date

if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
