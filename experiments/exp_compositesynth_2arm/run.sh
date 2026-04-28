#!/bin/bash
# Two-arm composite-synth A/B vs the dualemb_3arm baseline.
#
# Identical recipe to exp_dualemb_3arm except:
#   --synth-kind composite      (TimesFM-style: trend + ARIMA + 2 free waves
#                                + 1 seasonality-tied wave; per-row labels
#                                emitted with seas_id=0 when seas-tied off)
# Two arms (the EWMA-512 third arm is dropped — span=128 won GM-MASE on
# dualemb_3arm and is the headline choice):
#   Arm A: RevIN
#   Arm B: RevEWMNorm span=128
#
# Shared knobs:
#   * 30k bb + 30k qhead, batch 24, lr 1e-4 (bb) / 3e-4 (qh)
#   * Loss: cosine_similarity_batch
#   * mix_ratio 0.5  (50% bundle base_mixed_v1 + 50% on-the-fly composite)
#   * freq_emb_dim 3  +  seasonality_emb_dim 3
#   * mixup_p 0.3
#   * Selector: _best_loss → FINAL.pth
#
# Eval: full GIFT-Eval (97 configs) with the official evaluator. The
# backbone reads dataset.freq + get_seasonality(dataset.freq) per task
# and tags itself with the matching (freq_id, seasonality_id).
set -e
cd /workspace/app
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_compositesynth_2arm: starting ===" && date

# ----- Setup (idempotent — re-runs check before installing) ---------------
SETUP_MARKER=/workspace/app/.setup_done_compositesynth
if [ ! -f "$SETUP_MARKER" ]; then
    echo "=== SETUP ===" && date
    apt-get update -qq
    apt-get install -y -qq python3-pip rsync > /dev/null 2>&1 || true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "GPU: ${GPU_NAME} | driver: ${DRIVER_CUDA}"

    # torch is preinstalled in the pytorch/pytorch image; re-install will be
    # a no-op if the version pin matches. Force cu128 build.
    pip install --break-system-packages "torch>=2.8,<2.9" \
        --index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1 || true
    pip install --break-system-packages 'numpy<2' pandas pyarrow statsmodels \
        matplotlib datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
    pip install --break-system-packages \
        "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" \
        > /dev/null 2>&1

    python3 -c "import torch; print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

    # Pre-cache GIFT-Eval data once so all evals can find it.
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
SYNTH_KIND="composite"

run_backbone() {
    local NAME=$1; shift
    local NORM_KIND=$1; shift
    local NORM_SPAN=$1; shift
    local SPAN_FLAG=""
    if [ "$NORM_KIND" = "ewma" ]; then
        SPAN_FLAG="--rev-norm-span $NORM_SPAN"
    fi
    python3 -u experiments/freq-embedding/scripts/train.py \
        --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --save-every 2000 --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        --mix-ratio 0.5 --synth-kind "$SYNTH_KIND" \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
        --loss-shape "$LOSS"
    cp -f "checkpoints/${NAME}_best_loss.pth" "checkpoints/${NAME}_FINAL.pth"
}

run_qhead() {
    local QNAME=$1; shift
    local BB=$1; shift
    local NORM_KIND=$1; shift
    local NORM_SPAN=$1; shift
    local SPAN_FLAG=""
    if [ "$NORM_KIND" = "ewma" ]; then
        SPAN_FLAG="--rev-norm-span $NORM_SPAN"
    fi
    python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-every 1000 --save-dir checkpoints --run-name "$QNAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        --mix-ratio 0.5 --synth-kind "$SYNTH_KIND" \
        --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
        --reconstruction forecaster
    cp -f "checkpoints/${QNAME}_best.pth" "checkpoints/${QNAME}_FINAL.pth"
}

run_gift_eval() {
    local OUT_DIR=$1; shift
    local BB=$1; shift
    local QH=$1; shift
    local NORM_KIND=$1; shift
    local NORM_SPAN=$1; shift
    local SPAN_FLAG=""
    if [ "$NORM_KIND" = "ewma" ]; then
        SPAN_FLAG="--rev-norm-span $NORM_SPAN"
    fi
    python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
        --backbone-path "$BB" --head-path "$QH" \
        --output-dir "$OUT_DIR" --strategy B4 --forecast-len 16 \
        --rev-norm-kind "$NORM_KIND" $SPAN_FLAG --device cuda
}

mkdir -p experiments/exp_compositesynth_2arm/results

# ===== Arm A: RevIN =====
ABB="tiny_compsyn_revin"
AQH="R1q_compsyn_revin"
echo "" && echo "=== ARM A STAGE B: $ABB (RevIN) ===" && date
run_backbone "$ABB" "revin" 0
echo "=== ARM A STAGE B DONE ===" && date

echo "" && echo "=== ARM A STAGE H: $AQH ===" && date
run_qhead "$AQH" "checkpoints/${ABB}_FINAL.pth" "revin" 0
echo "=== ARM A STAGE H DONE ===" && date

echo "" && echo "=== ARM A STAGE E: gift_eval ===" && date
run_gift_eval "experiments/exp_compositesynth_2arm/results/gift_eval_revin" \
    "checkpoints/${ABB}_FINAL.pth" "checkpoints/${AQH}_FINAL.pth" "revin" 0
echo "=== ARM A STAGE E DONE ===" && date

# ===== Arm B: RevEWMNorm span=128 =====
BBB="tiny_compsyn_ewma128"
BQH="R1q_compsyn_ewma128"
echo "" && echo "=== ARM B STAGE B: $BBB (ewma span=128) ===" && date
run_backbone "$BBB" "ewma" 128
echo "=== ARM B STAGE B DONE ===" && date

echo "" && echo "=== ARM B STAGE H: $BQH ===" && date
run_qhead "$BQH" "checkpoints/${BBB}_FINAL.pth" "ewma" 128
echo "=== ARM B STAGE H DONE ===" && date

echo "" && echo "=== ARM B STAGE E: gift_eval ===" && date
run_gift_eval "experiments/exp_compositesynth_2arm/results/gift_eval_ewma128" \
    "checkpoints/${BBB}_FINAL.pth" "checkpoints/${BQH}_FINAL.pth" "ewma" 128
echo "=== ARM B STAGE E DONE ===" && date

echo "" && echo "=== run_compositesynth_2arm: ALL DONE ===" && date
echo ""
echo "GIFT-Eval summaries:"
for d in revin ewma128; do
    if [ -f "experiments/exp_compositesynth_2arm/results/gift_eval_${d}/summary.txt" ]; then
        echo "--- ${d} ---"
        head -20 "experiments/exp_compositesynth_2arm/results/gift_eval_${d}/summary.txt"
    fi
done
