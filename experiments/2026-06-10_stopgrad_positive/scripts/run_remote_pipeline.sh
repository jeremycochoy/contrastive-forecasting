#!/bin/bash
# Stop-grad follow-up (#339) — full pipeline on a single vast.ai RTX 4090:
#   setup (deps + GIFT-Eval data) → backbone 12.5k → downstream 2L → downstream 6L.
# Code is expected at /workspace/app (rsync'd from elisa), output goes to
# /workspace/out (runs/ + results/), everything tee'd to /workspace/out/run_all.log
# for the elisa sync loop. Stages are idempotent (skip-if-FINAL-exists), so
# re-running this script after a crash resumes where it left off.
set -uo pipefail
export WT=/workspace/app
export OUT=/workspace/out
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run_all.log") 2>&1

TAG="allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024"
SCRIPTS="$WT/experiments/2026-06-10_stopgrad_positive/scripts"

echo "=== SETUP ===" && date
apt-get update -qq 2>/dev/null
apt-get install -y -qq python3-pip rsync > /dev/null 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if ! python3 -c "import torch" 2>/dev/null; then
  echo "installing deps (torch cu128 + data/eval stack)"
  pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1
  pip3 install --break-system-packages 'numpy<2' pandas pyarrow statsmodels datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
  pip3 install --break-system-packages "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" > /dev/null 2>&1
fi
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch {torch.__version__}, CUDA OK, device: {torch.cuda.get_device_name(0)}')
" || { echo "FAILED: CUDA unavailable"; exit 1; }

export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt" 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "FAILED: missing $WT/experiments/hf_token.txt"; exit 1; }

# Primary: /workspace/gift-eval-data is scp'd from elisa during the push step
# (bit-identical to the reference arm's eval data). HF fallback only.
if [ ! -d /workspace/gift-eval-data ]; then
  echo "=== Download GIFT-Eval data (HF fallback) ===" && date
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
" || { echo "FAILED: GIFT-Eval data download"; exit 1; }
fi
export GIFT_EVAL=/workspace/gift-eval-data

echo "=== STAGE 1: backbone (12.5k, stop-grad) ===" && date
bash "$SCRIPTS/train_backbone_sgpos.sh" 0 12500 2500 || { echo "STAGE 1 FAILED"; exit 1; }

echo "=== STAGE 2: downstream 2L (best + last) ===" && date
bash "$SCRIPTS/downstream_sgpos.sh" "$TAG" 2 0 || { echo "STAGE 2 FAILED"; exit 1; }

echo "=== STAGE 3: downstream 6L (best + last) ===" && date
bash "$SCRIPTS/downstream_sgpos.sh" "$TAG" 6 0 || { echo "STAGE 3 FAILED"; exit 1; }

echo "=== ALL STAGES COMPLETE ===" && date
