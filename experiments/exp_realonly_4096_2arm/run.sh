#!/bin/bash
# exp_realonly_4096_2arm — real-data-only training on gift-pretrain-small-4096
#
# Purpose: train the contrastive backbone on T=4096, C=1 single-channel
# real data with NO synthetic mix (mix_ratio=0.0). Answers the question
# "how much of phases 1–5's gain came from synth vs real-data pretraining?"
# Also serves as the basis for upcoming architecture-search experiments.
#
# Dataset: jeremycochoy/gift-pretrain-small-4096 (small_v1 split, 61717
# rows × T=4096 windows × C=1, 32 parquet shards, 2.4 GB).
# At bs=24, one epoch = 2572 steps; total_steps=30k → ~11.7 epochs.
#
# Two arms with the per-norm best from phases 1–5:
#   - ewma128 arm: rev-norm-kind=ewma, span=128 (the v3-prim+EWMA winner's norm)
#   - revin   arm: rev-norm-kind=revin       (the v2pulse+RevIN winner's norm)
# Both use mix_ratio=0.0 (real data only, NO synth recipe).
#
# Usage:
#   bash experiments/exp_realonly_4096_2arm/run.sh ewma128
#   bash experiments/exp_realonly_4096_2arm/run.sh revin
set -e
cd /workspace/app

ARM="${1:?usage: run.sh <ewma128|revin>}"
case "$ARM" in
    ewma128)  NORM_KIND="ewma";   NORM_SPAN="128"; ;;
    revin)    NORM_KIND="revin";  NORM_SPAN="0";   ;;
    *) echo "Unknown arm: $ARM" >&2; exit 1 ;;
esac

LOG="/workspace/app/run_${ARM}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_realonly_4096: ARM ${ARM} starting ===" && date

SETUP_MARKER="/workspace/app/.setup_done_realonly_4096"
if [ ! -f "$SETUP_MARKER" ]; then
    echo "=== SETUP ===" && date
    apt-get update -qq
    apt-get install -y -qq python3-pip rsync > /dev/null 2>&1 || true

    pip install --break-system-packages "torch>=2.8,<2.9" \
        --index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1 || true
    pip install --break-system-packages 'numpy<2' pandas pyarrow statsmodels \
        matplotlib datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
    pip install --break-system-packages \
        "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" \
        > /dev/null 2>&1
    python3 -c "import torch; print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

    if [ ! -d /workspace/gift-eval-data ] || [ -z "$(ls -A /workspace/gift-eval-data 2>/dev/null)" ]; then
        echo "=== Download GIFT-Eval data ===" && date
        export HF_TOKEN_TMP=$(cat experiments/hf_token.txt)
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_TMP"
        # Download the eval/ split from gift-pretrain-small-4096 (it's the
        # standard GIFT-Eval test set, identical to the one in the bundles).
        python3 -c "
from huggingface_hub import snapshot_download
import os, shutil
path = snapshot_download('jeremycochoy/gift-pretrain-small-4096',
                         repo_type='dataset', allow_patterns='eval/**',
                         local_dir='/workspace/gift-eval-download')
src = os.path.join(path, 'eval')
dst = '/workspace/gift-eval-data'
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f'GIFT-Eval data ready: {dst}')
os.system(f'du -sh {dst}')
"
    fi
    touch "$SETUP_MARKER"
    echo "=== SETUP DONE ===" && date
fi

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/workspace/gift-eval-data

HF_REPO="jeremycochoy/gift-pretrain-small-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

SPAN_FLAG=""
if [ "$NORM_KIND" = "ewma" ]; then
    SPAN_FLAG="--rev-norm-span $NORM_SPAN"
fi

BB="tiny_realonly_4096_${ARM}"
QH="R1q_realonly_4096_${ARM}"
RES_DIR="experiments/exp_realonly_4096_2arm/results/gift_eval_${ARM}"
mkdir -p "$RES_DIR"

# ===== Backbone =====
# 30k steps, bs=24, T=4096, C=1, mix_ratio=0.0 (NO synth).
# At 61717 rows / 24 per step = 2572 steps/epoch → 30k = ~11.7 passes.
# Save every 2500 steps (~ once per epoch). The trainer auto-handles
# StopIteration → re-iter (multi-epoch loop is in-place).
echo "" && echo "=== ARM ${ARM} STAGE B: $BB (T=4096, C=1, mix=0.0) ===" && date
# NOTE: deliberately NO --grad-clip. The original NaN at step 1697 was a
# float32 cumsum overflow in RevEWMNorm at T=4096 (a real numerical
# defect), fixed by promoting the cumsum trick to float64 for T>2048
# (commit 452d79b). Grad-clip would mask future spikes and hide design
# issues we want to see — keeping training transparent.
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
    --save-every 2500 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
    --loss-shape "$LOSS"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== ARM ${ARM} STAGE B DONE ===" && date

# ===== qhead =====
# Forecasting head is trained on the same dataset shape (T=4096, C=1).
# mix_ratio still 0.0 here.
echo "" && echo "=== ARM ${ARM} STAGE H: $QH ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QH" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --mix-ratio 0.0 \
    --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
    --reconstruction forecaster
cp -f "checkpoints/${QH}_best.pth" "checkpoints/${QH}_FINAL.pth"
echo "=== ARM ${ARM} STAGE H DONE ===" && date

# ===== eval =====
# Eval uses the standard 97 GIFT-Eval configs at forecast_len=16 with the
# B4 strategy. SN-normalized MAPE/CRPS columns also emitted (per task #18).
# Need --t-raw 4096 --backbone-c 1 to match the trained backbone.
echo "" && echo "=== ARM ${ARM} STAGE E: gift_eval ===" && date
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --rev-norm-kind "$NORM_KIND" $SPAN_FLAG --device cuda
echo "=== ARM ${ARM} STAGE E DONE ===" && date

echo "" && echo "=== run_realonly_4096: ARM ${ARM} ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
