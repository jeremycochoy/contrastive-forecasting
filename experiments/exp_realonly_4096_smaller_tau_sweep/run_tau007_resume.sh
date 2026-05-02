#!/bin/bash
# Recovery launcher for τ=0.07 arm of #27 after the original machine3
# (35970433) was host-stopped at backbone step 2k on 2026-05-01 ~19:37.
# We push the locally-synced 2k checkpoint pair to /workspace/app/resume_source/
# (separate dir so safe_run_name doesn't auto-branch the run name) and resume
# Stage B from there; Stages H and E run as normal.
set -e
cd /workspace/app

ARM="007"
TAU="0.07"
LOG="/workspace/app/run_tau${ARM}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_tau007_resume: tau=${TAU} (arm=${ARM}) RESUMING ===" && date

SETUP_MARKER="/workspace/app/.setup_done_realonly_4096_smaller"
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

HF_REPO="jeremycochoy/gift-pretrain-small-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

BB="tiny_realonly_4096_smaller_tau${ARM}"
QH="R1q_realonly_4096_smaller_tau${ARM}"
RES_DIR="experiments/exp_realonly_4096_smaller_tau_sweep/results/gift_eval_tau${ARM}"
mkdir -p "$RES_DIR"
mkdir -p checkpoints

# Resume source (pre-pushed from local sync, plus more recent best_loss
# moved here after credit-restore). Kept in a separate directory so
# safe_run_name() doesn't see existing run-name checkpoints and branch us
# to "..._r2"; new checkpoints land cleanly in checkpoints/<run_name>_*.pth.
# Prefer 3600 (latest) if present, else fall back to 2k.
if [ -f "/workspace/app/resume_source/${BB}_3600.pth" ]; then
    RESUME_BB="/workspace/app/resume_source/${BB}_3600.pth"
else
    RESUME_BB="/workspace/app/resume_source/${BB}_2k.pth"
fi
if [ ! -f "$RESUME_BB" ]; then
    echo "ERROR: resume checkpoint missing at $RESUME_BB. Push it before running this script." >&2
    exit 2
fi

echo "" && echo "=== ARM ${ARM} STAGE B (RESUME from 2k): $BB (smaller, EWMA-128, tau=${TAU}, bs=96) ===" && date
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 96 --lr 1e-4 \
    --save-every 2500 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau "$TAU" \
    --loss-shape "$LOSS" \
    --resume "$RESUME_BB"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== ARM ${ARM} STAGE B DONE ===" && date

echo "" && echo "=== ARM ${ARM} STAGE H: $QH ===" && date
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
echo "=== ARM ${ARM} STAGE H DONE ===" && date

echo "" && echo "=== ARM ${ARM} STAGE E: gift_eval ===" && date
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== ARM ${ARM} STAGE E DONE ===" && date

echo "" && echo "=== run_tau007_resume: tau=${TAU} (arm=${ARM}) ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
