#!/bin/bash
# Periodic-synth-mix pipeline on a single vast.ai 4090 (or equivalent).
#
#   Stage 1a — backbone CONTROL: tiny_v3c_ctrl, 30k steps, pure base-bundles.
#   Stage 1b — backbone MIX:     tiny_v3c_mix,  30k steps, 50% base + 50% periodic synth.
#   Stage 2a — head R1v3c_ctrl on CONTROL backbone (30k forecaster recon).
#   Stage 2b — head R1v3c_mix  on MIX backbone.
#   Stage 3a — GIFT-Eval B4 on R1v3c_ctrl.
#   Stage 3b — GIFT-Eval B4 on R1v3c_mix.
#
# Output goes to run_all.log (tee'd) so the local sync loop can read progress.

set -e
cd /workspace/app

exec > >(tee -a /workspace/app/run_all.log) 2>&1

echo "=== SETUP ===" && date

apt-get update -qq
apt-get install -y -qq python3-pip rsync > /dev/null 2>&1

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU: ${GPU_NAME} | driver: ${DRIVER_CUDA}"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
echo "Using cu128 wheels"

pip3 install --break-system-packages torch --index-url "${CUDA_INDEX}" > /dev/null 2>&1
pip3 install --break-system-packages 'numpy<2' pandas pyarrow statsmodels datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
pip3 install --break-system-packages "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" > /dev/null 2>&1

# Fail fast if CUDA isn't actually available (spares us from downloading 1.5G
# of GIFT-Eval data onto a box that can't train).
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch {torch.__version__}, CUDA OK, device: {torch.cuda.get_device_name(0)}')
"

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0

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

mkdir -p checkpoints results

# ============================================================
# STAGE 1a — backbone CONTROL (pure base-bundles, 30k steps)
# ============================================================
echo "" && echo "=== STAGE 1a: backbone CONTROL (tiny_v3c_ctrl) ===" && date
python3 -u experiments/2026-04-27_periodic-synth-mix/scripts/train.py \
    --device cuda \
    --total-steps 30000 \
    --batch-size 24 \
    --lr 1e-4 \
    --save-dir checkpoints \
    --run-name tiny_v3c_ctrl \
    --hf-repo jeremycochoy/contrastive-training-base-bundles \
    --hf-path base_mixed_v1 \
    --mix-ratio 0.0 \
    --seed 42
echo "=== STAGE 1a DONE ===" && date

# ============================================================
# STAGE 1b — backbone MIX (50/50 base + periodic synth, 30k steps)
# ============================================================
echo "" && echo "=== STAGE 1b: backbone MIX (tiny_v3c_mix) ===" && date
python3 -u experiments/2026-04-27_periodic-synth-mix/scripts/train.py \
    --device cuda \
    --total-steps 30000 \
    --batch-size 24 \
    --lr 1e-4 \
    --save-dir checkpoints \
    --run-name tiny_v3c_mix \
    --hf-repo jeremycochoy/contrastive-training-base-bundles \
    --hf-path base_mixed_v1 \
    --mix-ratio 0.5 \
    --seed 42
echo "=== STAGE 1b DONE ===" && date

# Pick best_gap checkpoint for each backbone.
CTRL_BACKBONE=$(ls -t checkpoints/tiny_v3c_ctrl*_best_gap.pth 2>/dev/null | grep -v optimizer | head -1)
MIX_BACKBONE=$(ls -t checkpoints/tiny_v3c_mix*_best_gap.pth 2>/dev/null | grep -v optimizer | head -1)
echo "CTRL backbone: $CTRL_BACKBONE"
echo "MIX  backbone: $MIX_BACKBONE"
test -s "$CTRL_BACKBONE"
test -s "$MIX_BACKBONE"

# ============================================================
# STAGE 2a — R1 head on CONTROL backbone
# ============================================================
echo "" && echo "=== STAGE 2a: head R1v3c_ctrl ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "$CTRL_BACKBONE" \
    --forecast-len 16 \
    --reconstruction forecaster \
    --total-steps 30000 \
    --batch-size 24 \
    --lr 3e-4 \
    --save-dir checkpoints \
    --run-name R1v3c_ctrl \
    --hf-repo jeremycochoy/contrastive-training-base-bundles \
    --hf-path base_mixed_v1 \
    --device cuda
echo "=== STAGE 2a DONE ===" && date

# ============================================================
# STAGE 2b — R1 head on MIX backbone
# ============================================================
echo "" && echo "=== STAGE 2b: head R1v3c_mix ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "$MIX_BACKBONE" \
    --forecast-len 16 \
    --reconstruction forecaster \
    --total-steps 30000 \
    --batch-size 24 \
    --lr 3e-4 \
    --save-dir checkpoints \
    --run-name R1v3c_mix \
    --hf-repo jeremycochoy/contrastive-training-base-bundles \
    --hf-path base_mixed_v1 \
    --device cuda
echo "=== STAGE 2b DONE ===" && date

# ============================================================
# STAGE 3a — GIFT-Eval B4 on CONTROL
# ============================================================
echo "" && echo "=== STAGE 3a: GIFT-Eval R1v3c_ctrl ===" && date
mkdir -p results/R1v3c_ctrl
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "$CTRL_BACKBONE" \
    --head-path checkpoints/R1v3c_ctrl_best.pth \
    --forecast-len 16 \
    --strategy B4 \
    --output-dir results/R1v3c_ctrl \
    --device cuda
echo "=== STAGE 3a DONE ===" && date

# ============================================================
# STAGE 3b — GIFT-Eval B4 on MIX
# ============================================================
echo "" && echo "=== STAGE 3b: GIFT-Eval R1v3c_mix ===" && date
mkdir -p results/R1v3c_mix
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "$MIX_BACKBONE" \
    --head-path checkpoints/R1v3c_mix_best.pth \
    --forecast-len 16 \
    --strategy B4 \
    --output-dir results/R1v3c_mix \
    --device cuda
echo "=== STAGE 3b DONE ===" && date

echo "" && echo "=== ALL STAGES COMPLETE ===" && date
ls -la results/R1v3c_ctrl results/R1v3c_mix
echo ""
echo "CONTROL first rows:"; head -5 results/R1v3c_ctrl/all_results.csv 2>/dev/null || echo "(missing)"
echo "MIX first rows:";     head -5 results/R1v3c_mix/all_results.csv 2>/dev/null || echo "(missing)"
