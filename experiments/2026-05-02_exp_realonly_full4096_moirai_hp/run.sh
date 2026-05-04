#!/bin/bash
# 2026-05-02_exp_realonly_full4096_moirai_hp (#9 / original #31) — same configuration as
# #6 (30k learnable-τ baseline on gift-pretrain-full-4096) but with MOIRAI
# optimizer hyperparameters per Aksu et al.: AdamW lr=1e-3, weight_decay=0.1,
# β1=0.9, β2=0.98 (vs default lr=1e-4, wd=0.01, β2=0.999). NO warmup, NO
# cosine annealing — flat lr to keep the comparison clean against #6.
#
# Goal: directly isolate the optimizer-HP axis by running the SAME arch +
# τ-policy + dataset + step budget as #6, only optimizer changes.
#
# Dataset:  jeremycochoy/gift-pretrain-full-4096 (path=small_v1; full data
#           lives there per dataset README naming convention).
# Steps:    30k backbone + 30k qhead + STAGE E gift_eval — mirrors #6 budget
#           so cross-experiment comparison is apples-to-apples.
# Arch:    smaller (L=6 H=384 nhead=6, 11.4M), EWMA-128, bs=96, T=4096, C=1,
#           mix_ratio=0.0.
# τ-policy: --tau 0.07 --learnable-tau (same as #6).
# Optim:    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
# Anti:     NO grad-clip (banned project-wide).
#
# Watch out: 10× lr is a real instability risk; watch the first 1k steps
# for NaN. Per project rule, fix the data/normalization if NaN — never
# add grad-clip.
set -e
cd /workspace/app

LOG="/workspace/app/run_full4096_moirai_hp.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_full4096_moirai_hp: starting ===" && date

SETUP_MARKER="/workspace/app/.setup_done_realonly_full4096"
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
        export HF_TOKEN_TMP=$(cat experiments/hf_token.txt)
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_TMP"
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
"
    fi
    touch "$SETUP_MARKER"
fi

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/workspace/gift-eval-data

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

BB="tiny_realonly_full4096_moirai_hp"
QH="R1q_realonly_full4096_moirai_hp"
RES_DIR="experiments/2026-05-02_exp_realonly_full4096_moirai_hp/results/gift_eval"
mkdir -p "$RES_DIR"

echo "" && echo "=== STAGE B: $BB (smaller, EWMA-128, learnable τ, bs=96, MOIRAI HP: lr=1e-3 wd=0.1 β2=0.98) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 96 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 2500 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.07 --learnable-tau \
    --loss-shape "$LOSS"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== STAGE B DONE ===" && date

echo "" && echo "=== STAGE H: $QH (qhead trained with default optim — MOIRAI HP only on backbone) ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 96 --lr 3e-4 \
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
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== STAGE E DONE ===" && date

echo "" && echo "=== run_full4096_moirai_hp: ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
