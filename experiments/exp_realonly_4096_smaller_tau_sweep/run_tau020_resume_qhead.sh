#!/bin/bash
# Recovery launcher for τ=0.20 arm (#27) after credit-restore: qhead was at
# step ~23k of 30k when the account went negative. We resume the head trainer
# from R1q_*_best.pth (which has its companion _optimizer.pth alongside),
# finish the remaining steps, then run gift-eval.
set -e
cd /workspace/app

ARM="020"
LOG="/workspace/app/run_tau${ARM}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_tau020_resume_qhead: arm=${ARM} after credit-restore ===" && date

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

BB="tiny_realonly_4096_smaller_tau${ARM}"
QH="R1q_realonly_4096_smaller_tau${ARM}"
RES_DIR="experiments/exp_realonly_4096_smaller_tau_sweep/results/gift_eval_tau${ARM}"
mkdir -p "$RES_DIR"

RESUME_HEAD="/workspace/app/checkpoints/${QH}_best.pth"
if [ ! -f "$RESUME_HEAD" ]; then
    echo "ERROR: head resume checkpoint missing at $RESUME_HEAD" >&2
    exit 2
fi

echo "" && echo "=== STAGE H (RESUME): $QH ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 96 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QH" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster \
    --resume "$RESUME_HEAD"
cp -f "checkpoints/${QH}_best.pth" "checkpoints/${QH}_FINAL.pth"
echo "=== STAGE H DONE ===" && date

echo "" && echo "=== STAGE E (RESUME if partial CSV exists): gift_eval tau${ARM} ===" && date
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --resume
echo "=== STAGE E DONE ===" && date

echo "" && echo "=== run_tau020_resume_qhead: tau=0.20 (arm=${ARM}) ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
