#!/bin/bash
# Phase 5: bump exp envelope total-gain range from (0.1, 10) → (0.01, 100).
# Goal: expose covid-style 100× explosive growth/decay in synth.
# Using best-of-breed flags per norm:
#   - revin   arm: --enable-pulse + --env-gain-max 100  (tests env-bump on top of v2pulse winner)
#   - ewma128 arm: --more-primitives + --env-gain-max 100  (tests env-bump on top of v3 winner)
#
# Usage:
#   bash experiments/exp_compositesynth_v5envboost_2arm/run.sh revin
#   bash experiments/exp_compositesynth_v5envboost_2arm/run.sh ewma128
set -e
cd /workspace/app

ARM="${1:?usage: run.sh <revin|ewma128>}"
case "$ARM" in
    revin)    NORM_KIND="revin";   NORM_SPAN="0";   EXTRA_FLAGS="--enable-pulse"; ;;
    ewma128)  NORM_KIND="ewma";    NORM_SPAN="128"; EXTRA_FLAGS="--more-primitives"; ;;
    *) echo "Unknown arm: $ARM" >&2; exit 1 ;;
esac

LOG="/workspace/app/run_${ARM}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_compositesynth_v5envboost: ARM ${ARM} starting ===" && date

SETUP_MARKER="/workspace/app/.setup_done_compositesynth_v5"
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
path = snapshot_download('jeremycochoy/contrastive-training-tiny-bundles',
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

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"
LOSS="cosine_similarity_batch"

SPAN_FLAG=""
if [ "$NORM_KIND" = "ewma" ]; then
    SPAN_FLAG="--rev-norm-span $NORM_SPAN"
fi

BB="tiny_compsyn_v5_${ARM}"
QH="R1q_compsyn_v5_${ARM}"
RES_DIR="experiments/exp_compositesynth_v5envboost_2arm/results/gift_eval_${ARM}"
mkdir -p "$RES_DIR"

# ===== Backbone =====
echo "" && echo "=== ARM ${ARM} STAGE B: $BB (env-gain-max 100, ${EXTRA_FLAGS}) ===" && date
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
    --save-every 2000 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --mix-ratio 0.5 --synth-kind composite ${EXTRA_FLAGS} --env-gain-max 100 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
    --loss-shape "$LOSS"
cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== ARM ${ARM} STAGE B DONE ===" && date

# ===== qhead =====
echo "" && echo "=== ARM ${ARM} STAGE H: $QH ===" && date
python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QH" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --mix-ratio 0.5 --synth-kind composite ${EXTRA_FLAGS} --env-gain-max 100 \
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
    --rev-norm-kind "$NORM_KIND" $SPAN_FLAG --device cuda
echo "=== ARM ${ARM} STAGE E DONE ===" && date

echo "" && echo "=== run_compositesynth_v5envboost: ARM ${ARM} ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
