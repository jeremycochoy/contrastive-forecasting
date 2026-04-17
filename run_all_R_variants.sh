#!/bin/bash
# Full pipeline: R1 eval → R2 train+eval → R3 train+eval → R4 train+eval
# Run on RTX 5090 with cu130 torch
set -e
cd /workspace/app
export PYTHONPATH=/workspace/app
export GIFT_EVAL=/workspace/gift-eval-data
export CUDA_VISIBLE_DEVICES=0

echo "================================================================"
echo "=== R-VARIANT PIPELINE START ===" && date
echo "================================================================"

# Step 0: Ensure all deps are installed
echo "=== Installing deps ===" && date
pip3 install --break-system-packages "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" 2>&1 | tail -3

# Step 0b: Download GIFT-Eval data from HF (needed for evals)
echo "=== Downloading GIFT-Eval data ===" && date
python3 -c "
from huggingface_hub import snapshot_download
import os, shutil
path = snapshot_download('jeremycochoy/contrastive-training-tiny-bundles', repo_type='dataset', allow_patterns='eval/**', local_dir='/workspace/gift-eval-download')
src = os.path.join(path, 'eval')
dst = '/workspace/gift-eval-data'
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f'Data ready: {dst}')
os.system(f'du -sh {dst}')
"

# ============================================================
# R1: forecaster reconstruction W=16 (already trained)
# ============================================================
echo "=== R1 eval (forecaster recon W=16) ===" && date
mkdir -p results/R1
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --head-path checkpoints/R1_forecaster_recon_w16_best.pth \
    --forecast-len 16 --strategy B4 \
    --output-dir results/R1 --device cuda
echo "=== R1 eval DONE ===" && date

# ============================================================
# R2: encoder reconstruction W=16
# ============================================================
echo "=== R2 train (encoder recon W=16) ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --forecast-len 16 --reconstruction encoder \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-dir checkpoints --run-name R2_encoder_recon_w16 \
    --hf-repo jeremycochoy/contrastive-training-tiny-bundles \
    --hf-path tiny_mixed_v2 --device cuda
echo "=== R2 train DONE ===" && date

echo "=== R2 eval ===" && date
mkdir -p results/R2
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --head-path checkpoints/R2_encoder_recon_w16_best.pth \
    --forecast-len 16 --strategy B4 \
    --output-dir results/R2 --device cuda
echo "=== R2 eval DONE ===" && date

# ============================================================
# R3: rolled reconstruction W=16 (mixed-rollout + reconstruction)
# ============================================================
echo "=== R3 train (rolled recon W=16) ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --forecast-len 16 --reconstruction forecaster --mixed-rollout 8 \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-dir checkpoints --run-name R3_rolled_recon_w16 \
    --hf-repo jeremycochoy/contrastive-training-tiny-bundles \
    --hf-path tiny_mixed_v2 --device cuda
echo "=== R3 train DONE ===" && date

echo "=== R3 eval ===" && date
mkdir -p results/R3
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --head-path checkpoints/R3_rolled_recon_w16_best.pth \
    --forecast-len 16 --strategy B4 \
    --output-dir results/R3 --device cuda
echo "=== R3 eval DONE ===" && date

# ============================================================
# R4: forecaster reconstruction W=128
# ============================================================
echo "=== R4 train (forecaster recon W=128) ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --forecast-len 128 --reconstruction forecaster \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-dir checkpoints --run-name R4_forecaster_recon_w128 \
    --hf-repo jeremycochoy/contrastive-training-tiny-bundles \
    --hf-path tiny_mixed_v2 --device cuda
echo "=== R4 train DONE ===" && date

echo "=== R4 eval ===" && date
mkdir -p results/R4
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path checkpoints/tiny_v2_best_gap.pth \
    --head-path checkpoints/R4_forecaster_recon_w128_best.pth \
    --forecast-len 128 --strategy B1 \
    --output-dir results/R4 --device cuda
echo "=== R4 eval DONE ===" && date

echo "================================================================"
echo "=== ALL R-VARIANTS DONE ===" && date
echo "================================================================"
# List all results
ls -la results/R*/all_results.csv 2>/dev/null
