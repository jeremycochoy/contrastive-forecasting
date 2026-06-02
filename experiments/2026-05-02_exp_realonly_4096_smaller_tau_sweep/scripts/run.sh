#!/bin/bash
# 2026-05-02_exp_realonly_4096_smaller_tau_sweep — τ sweep on smaller-EWMA-128 with
# bigger batch (bs=96 vs prior 24). 0.07 is the reference (already
# covered by exp_realonly_4096_smaller_2arm/ewma128). This script handles
# the new τ values.
#
# Usage:
#   bash experiments/2026-05-02_exp_realonly_4096_smaller_tau_sweep/run.sh 005    # τ=0.05
#   bash experiments/2026-05-02_exp_realonly_4096_smaller_tau_sweep/run.sh 020    # τ=0.20
set -e
cd /workspace/app

ARM="${1:?usage: run.sh <005|007|020>}"
case "$ARM" in
    005) TAU="0.05" ;;
    007) TAU="0.07" ;;  # bs=96 anchor (re-run since #20 was bs=24)
    020) TAU="0.20" ;;
    *) echo "Unknown arm: $ARM (expected 005, 007, or 020)" >&2; exit 1 ;;
esac

LOG="/workspace/app/run_tau${ARM}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_smaller_tau_sweep: tau=${TAU} (arm=${ARM}) starting ===" && date

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
RES_DIR="experiments/2026-05-02_exp_realonly_4096_smaller_tau_sweep/results/gift_eval_tau${ARM}"
mkdir -p "$RES_DIR"

# bs=96 chosen via local benchmark on smaller arch + T=4096:
#   bs=24:  4.31 GB peak    bs=96:  19.82 GB peak
#   bs=48:  8.47 GB peak    bs=128: OOM
#   bs=64:  11.63 GB peak
echo "" && echo "=== ARM ${ARM} STAGE B: $BB (smaller, EWMA-128, tau=${TAU}, bs=96) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 96 --lr 1e-4 \
    --save-every 2500 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau "$TAU" \
    --loss-shape "$LOSS"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== ARM ${ARM} STAGE B DONE ===" && date

echo "" && echo "=== ARM ${ARM} STAGE H: $QH ===" && date
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
echo "=== ARM ${ARM} STAGE H DONE ===" && date

echo "" && echo "=== ARM ${ARM} STAGE E: gift_eval ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== ARM ${ARM} STAGE E DONE ===" && date

echo "" && echo "=== run_smaller_tau_sweep: tau=${TAU} (arm=${ARM}) ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
