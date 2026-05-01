#!/bin/bash
# exp_realonly_4096_smaller_span_sweep — EWMA-128 span sweep on the
# smaller arch (L=6 H=384 nhead=6) under the realonly + T=4096 + C=1
# setting.
#
# Usage:
#   bash experiments/exp_realonly_4096_smaller_span_sweep/run.sh <span>
# Where <span> is one of 32, 64, 256, 512 (span=128 is already covered
# by exp_realonly_4096_smaller_2arm/ewma128).
set -e
cd /workspace/app

SPAN="${1:?usage: run.sh <span>}"
case "$SPAN" in
    32|64|256|512) ;;  # ok
    128) echo "span=128 already covered by exp_realonly_4096_smaller_2arm; refuse to redo"; exit 1 ;;
    *) echo "Unknown span: $SPAN (expected 32/64/256/512)" >&2; exit 1 ;;
esac

NORM_KIND="ewma"
ARM="ewma_span${SPAN}"

LOG="/workspace/app/run_span${SPAN}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_smaller_span_sweep: span=${SPAN} starting ===" && date

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

BB="tiny_realonly_4096_smaller_${ARM}"
QH="R1q_realonly_4096_smaller_${ARM}"
RES_DIR="experiments/exp_realonly_4096_smaller_span_sweep/results/gift_eval_${ARM}"
mkdir -p "$RES_DIR"

echo "" && echo "=== ARM ${ARM} STAGE B: $BB (L=6 H=384 nhead=6, span=${SPAN}) ===" && date
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
    --save-every 2500 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind "$NORM_KIND" --rev-norm-span "$SPAN" \
    --grad-clip 1.0 \
    --loss-shape "$LOSS"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== ARM ${ARM} STAGE B DONE ===" && date

echo "" && echo "=== ARM ${ARM} STAGE H: $QH ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QH" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind "$NORM_KIND" --rev-norm-span "$SPAN" \
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
    --rev-norm-kind "$NORM_KIND" --rev-norm-span "$SPAN" --device cuda
echo "=== ARM ${ARM} STAGE E DONE ===" && date

echo "" && echo "=== run_smaller_span_sweep: span=${SPAN} ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
