#!/bin/bash
# Continuation script for the RevIN arm of exp_realonly_4096_2arm.
# Used after the original revin instance died mid-pipeline (the backbone
# FINAL.pth survived locally). This script skips stage B (backbone) and
# runs only stage H (qhead) + stage E (eval), assuming the backbone FINAL
# is already present at checkpoints/tiny_realonly_4096_revin_FINAL.pth on
# the remote.
#
# Usage:
#   bash experiments/exp_realonly_4096_2arm/run_revin_resume.sh
set -e
cd /workspace/app

ARM="revin"
NORM_KIND="revin"
SPAN_FLAG=""

LOG="/workspace/app/run_revin.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_realonly_4096 (resume): ARM ${ARM} starting ===" && date

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

BB="tiny_realonly_4096_revin"
QH="R1q_realonly_4096_revin"
RES_DIR="experiments/exp_realonly_4096_2arm/results/gift_eval_${ARM}"
mkdir -p "$RES_DIR"

# ===== Verify backbone FINAL is in place (uploaded by orchestrator) =====
if [ ! -f "checkpoints/${BB}_FINAL.pth" ]; then
    echo "FATAL: backbone checkpoint missing at checkpoints/${BB}_FINAL.pth"
    echo "Upload it from the local sync dir before running this script."
    exit 1
fi
ls -la "checkpoints/${BB}_FINAL.pth"

# ===== qhead =====
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
echo "" && echo "=== ARM ${ARM} STAGE E: gift_eval ===" && date
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --rev-norm-kind "$NORM_KIND" $SPAN_FLAG --device cuda
echo "=== ARM ${ARM} STAGE E DONE ===" && date

echo "" && echo "=== run_realonly_4096 (resume): ARM ${ARM} ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
