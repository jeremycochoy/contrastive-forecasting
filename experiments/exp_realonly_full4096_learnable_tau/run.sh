#!/bin/bash
# exp_realonly_full4096_learnable_tau (#33 FINAL) — re-run the #27/#32
# winner config on the FULL gift-pretrain dataset.
#
# Winner declared on May 2 2026: learnable τ (CLIP-style log_inv_tau,
# init=0.07). #31 (MOIRAI optim sweep) skipped per user — we go straight
# to #33 with default optimizer hyperparams from #32.
#
# Dataset:    jeremycochoy/gift-pretrain-full-4096 (42.5M windows, 619 GB,
#             4274 zstd parquet shards). Per the dataset README, the
#             directory is named `small_v1/` for tooling parity with the
#             companion small dataset — it holds the FULL data despite
#             the name. Use --hf-path small_v1.
# Step count: 30k backbone + 30k qhead, mirroring #27/#32 budgets so
#             cross-experiment loss curves are directly comparable. At
#             bs=96 this consumes 2.88M samples = ~7% of one full epoch
#             ⇒ no row repeats ⇒ no memorization regime (the whole point
#             of moving from small to full).
# Arch:       smaller (L=6 H=384 nhead=6, 11.4M params), EWMA-128, bs=96,
#             T=4096, C=1, mix_ratio=0.0 — same as #20/#27/#32.
# τ-policy:   --tau 0.07 --learnable-tau (init τ=0.07, log_inv_tau is a
#             trainable scalar clamped to [log(1), log(100)] each step).
# Optimizer:  AdamW lr=1e-4, default weight_decay=0, default betas
#             (no MOIRAI hyperparams — those are deferred).
# Anti-rules: NO grad-clip (banned project-wide).
set -e
cd /workspace/app

LOG="/workspace/app/run_full4096_learnable.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_full4096_learnable_tau: starting ===" && date

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
    # GIFT-Eval data — pulled from the small dataset (same eval split,
    # bytes-identical mirror of Salesforce/GiftEval). Either repo's
    # eval/ subdir would do; small is faster to download.
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
HF_PATH="small_v1"  # not a typo — full dataset's dir is named small_v1 for parity
LOSS="cosine_similarity_batch"

BB="tiny_realonly_full4096_learnable_tau"
QH="R1q_realonly_full4096_learnable_tau"
RES_DIR="experiments/exp_realonly_full4096_learnable_tau/results/gift_eval"
mkdir -p "$RES_DIR"

echo "" && echo "=== STAGE B: $BB (smaller, EWMA-128, learnable τ init=0.07, bs=96, FULL dataset) ===" && date
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 96 --lr 1e-4 \
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

echo "" && echo "=== STAGE H: $QH ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
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
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== STAGE E DONE ===" && date

echo "" && echo "=== run_full4096_learnable_tau: ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
