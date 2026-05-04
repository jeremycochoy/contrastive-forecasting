#!/bin/bash
# v3b continuation on a single Vast.ai machine:
#   Stage 1: Resume v3b backbone training from tiny_v3b_r3_40k.pth -> 500k steps
#   Stage 2: Train R1v3b head (forecaster reconstruction, W=16, 30k steps)
#   Stage 3: GIFT-Eval on 97 configs (strategy B4, W=16)
#
# Assumes --resume-from uploaded /tmp/v3b_resume/checkpoints/ to /workspace/app/checkpoints/.
# Output log: tee'd to /workspace/app/run_all.log for the sync loop to pull.
set -e
cd /workspace/app

# Tee everything so the sync loop can read progress from a single file.
exec > >(tee -a /workspace/app/run_all.log) 2>&1

echo "=== SETUP ===" && date

apt-get update -qq
apt-get install -y -qq python3-pip rsync > /dev/null 2>&1

# cu128 works with CUDA 12.8 drivers (most modern vast machines) and supports
# Ada (4090) + Blackwell (5090) GPUs. Forward-compat of cu124 wheels against
# driver 565+ / CUDA 12.8 runtime fails with error 804 on some hosts.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU: ${GPU_NAME} | driver: ${DRIVER_CUDA}"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
echo "Using cu128 wheels"
pip3 install --break-system-packages torch --index-url "${CUDA_INDEX}" > /dev/null 2>&1
pip3 install --break-system-packages 'numpy<2' pandas pyarrow statsmodels datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
pip3 install --break-system-packages "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" > /dev/null 2>&1
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0

# Pull GIFT-Eval data upfront so Stage 3 has it ready.
# eval/ subfolder lives in tiny-bundles (base-bundles has no eval/).
echo "=== Download GIFT-Eval data ===" && date
python3 -c "
from huggingface_hub import snapshot_download
import os, shutil
path = snapshot_download('jeremycochoy/contrastive-training-tiny-bundles', repo_type='dataset', allow_patterns='eval/**', local_dir='/workspace/gift-eval-download')
src = os.path.join(path, 'eval')
dst = '/workspace/gift-eval-data'
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f'GIFT-Eval data ready: {dst}')
os.system(f'du -sh {dst}')
"
export GIFT_EVAL=/workspace/gift-eval-data

echo "=== Resume checkpoint verification ===" && date
# Pick highest numeric step suffix (e.g. _120k > _90k > _70k) regardless of rN prefix.
RESUME_CKPT=$(for f in checkpoints/tiny_v3b_*k.pth; do
    [[ "$f" == *optimizer* ]] && continue
    n=$(echo "$f" | sed -E 's/.*_([0-9]+)k\.pth/\1/')
    printf '%s %s\n' "$n" "$f"
done | sort -n | tail -1 | awk '{print $2}')
echo "Selected resume checkpoint: ${RESUME_CKPT}"
test -s "${RESUME_CKPT}"
test -s "${RESUME_CKPT%.pth}_optimizer.pth"

# ============================================================
# STAGE 1: v3b backbone training (40k -> 500k)
# ============================================================
echo "" && echo "=== STAGE 1: v3b backbone ===" && date
python3 -u experiments/2026-04-12_tiny-training/scripts/train.py \
    --device cuda \
    --total-steps 200000 \
    --batch-size 24 \
    --lr 1e-4 \
    --save-dir checkpoints \
    --run-name tiny_v3b \
    --hf-repo jeremycochoy/contrastive-training-base-bundles \
    --hf-path base_mixed_v1 \
    --resume "${RESUME_CKPT}"
echo "=== STAGE 1 DONE ===" && date
echo ""

BACKBONE=$(ls -t checkpoints/tiny_v3b*_best_gap.pth 2>/dev/null | grep -v optimizer | head -1)
echo "Backbone for head+eval: $BACKBONE"
test -s "$BACKBONE"

# ============================================================
# STAGE 2: R1v3b head (W=16 forecaster reconstruction)
# ============================================================
echo "" && echo "=== STAGE 2: R1v3b head ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "$BACKBONE" \
    --forecast-len 16 \
    --reconstruction forecaster \
    --total-steps 30000 \
    --batch-size 24 \
    --lr 3e-4 \
    --save-dir checkpoints \
    --run-name R1v3b \
    --hf-repo jeremycochoy/contrastive-training-base-bundles \
    --hf-path base_mixed_v1 \
    --device cuda
echo "=== STAGE 2 DONE ===" && date
echo ""

# ============================================================
# STAGE 3: GIFT-Eval (strategy B4, W=16)
# ============================================================
echo "" && echo "=== STAGE 3: GIFT-Eval ===" && date
mkdir -p results/R1v3b
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "$BACKBONE" \
    --head-path checkpoints/R1v3b_best.pth \
    --forecast-len 16 \
    --strategy B4 \
    --output-dir results/R1v3b \
    --device cuda
echo "=== STAGE 3 DONE ===" && date
echo ""

echo "=== ALL STAGES COMPLETE ===" && date
ls -la results/R1v3b/
echo ""
echo "First rows of all_results.csv:"
head -5 results/R1v3b/all_results.csv 2>/dev/null || echo "(missing)"
