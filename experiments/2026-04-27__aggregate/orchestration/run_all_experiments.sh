#!/bin/bash
# Multi-experiment driver for the late-Apr-2026 sequence on top of v3b.
# All four experiments run sequentially on a single 4090 instance.
#
# Stages:
#   EXP1  RevIN reproduction (#28 redo, lost-checkpoint replacement)
#   EXP2  Synth-only training (mix_ratio=1.0 backbones at 30k and 60k)
#   EXP3  RevEWMNorm span sweep (span ∈ {64, 128, 256}; span=32 reuses fe+mu locally)
#   EXP4  Patch-stats feature: per-patch RevEWMNorm diff stats injected into encoder
#
# Each stage prints a clear === EXPN STAGEM ... === marker so the local
# Monitor can track progress via grep alternation.
set -e
cd /workspace/app

# All output gets tee'd to run_all.log so the sync_loop pulls it.
exec > >(tee -a /workspace/app/run_all.log) 2>&1
echo "=== SETUP ===" && date

apt-get update -qq
apt-get install -y -qq python3-pip rsync > /dev/null 2>&1

# CUDA 12.8 wheels — works on any 4090 host with driver 565+.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU: ${GPU_NAME} | driver: ${DRIVER_CUDA}"
pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1
pip3 install --break-system-packages 'numpy<2' pandas pyarrow statsmodels datasets huggingface_hub tqdm gluonts > /dev/null 2>&1
pip3 install --break-system-packages "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git" > /dev/null 2>&1
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
# HF token — required to avoid throttling (CLAUDE.md rule). Token file
# is uploaded as part of the code tarball.
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "HF token: ${HF_TOKEN:0:8}…"

# Pre-cache GIFT-Eval data once so all evals can find it.
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

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"

run_train_backbone() {
    # $1=run_name, $2..$N=extra args (passed to train.py)
    local NAME=$1; shift
    python3 -u experiments/freq-embedding/scripts/train.py \
        --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        "$@"
}

run_qhead() {
    # $1=run_name, $2=backbone path, $3..=extra args
    local NAME=$1; shift
    local BB=$1; shift
    python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        "$@"
}

run_eval_full() {
    # $1=run_name, $2=backbone, $3=head, $4..=extra args
    local NAME=$1; shift
    local BB=$1; shift
    local HEAD=$1; shift
    mkdir -p "results/${NAME}"
    python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
        --backbone-path "$BB" --head-path "$HEAD" \
        --forecast-len 16 --strategy B4 \
        --output-dir "results/${NAME}" --device cuda \
        "$@"
}

# ============================================================
# EXP1 — RevIN reproduction (replaces lost #28 RevIN checkpoint)
# ============================================================
echo "" && echo "=== EXP1 STAGE 1: RevIN backbone (fe+mu+revin, 30k) ===" && date
run_train_backbone tiny_femu_revin \
    --mix-ratio 0.5 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind revin
echo "=== EXP1 STAGE 1 DONE ===" && date

# Permanent-name the best-gap backbone immediately (Checkpoint Safety Rule #1).
cp -f checkpoints/tiny_femu_revin_best_gap.pth checkpoints/tiny_femu_revin_FINAL.pth

echo "" && echo "=== EXP1 STAGE 2: RevIN qhead (30k) ===" && date
run_qhead R1q_femu_revin_v2 checkpoints/tiny_femu_revin_FINAL.pth \
    --rev-norm-kind revin
echo "=== EXP1 STAGE 2 DONE ===" && date
cp -f checkpoints/R1q_femu_revin_v2_best.pth checkpoints/R1q_femu_revin_v2_FINAL.pth

# ============================================================
# EXP2 — Synth-only training (mix_ratio=1.0)
# ============================================================
echo "" && echo "=== EXP2 STAGE 1: synth-only backbone 30k ===" && date
run_train_backbone tiny_femu_synthonly_30k \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 --rev-norm-kind ewma
echo "=== EXP2 STAGE 1 DONE ===" && date
cp -f checkpoints/tiny_femu_synthonly_30k_best_gap.pth \
      checkpoints/tiny_femu_synthonly_30k_FINAL.pth

echo "" && echo "=== EXP2 STAGE 2: synth-only backbone 60k ===" && date
python3 -u experiments/freq-embedding/scripts/train.py \
    --device cuda --total-steps 60000 --batch-size 24 --lr 1e-4 \
    --save-dir checkpoints --run-name tiny_femu_synthonly_60k \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 --rev-norm-kind ewma
echo "=== EXP2 STAGE 2 DONE ===" && date
cp -f checkpoints/tiny_femu_synthonly_60k_best_gap.pth \
      checkpoints/tiny_femu_synthonly_60k_FINAL.pth

# Quantile heads on synth-only data — same architecture as existing
# fe+mu+qh, just trained on synth-only. The synth-grid plot will then
# answer the open question: does stripping the real-data half help the
# head reproduce its training distribution?
echo "" && echo "=== EXP2 STAGE 3: synth-only qhead 30k ===" && date
run_qhead R1q_femu_synthonly_30k checkpoints/tiny_femu_synthonly_30k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma
echo "=== EXP2 STAGE 3 DONE ===" && date
cp -f checkpoints/R1q_femu_synthonly_30k_best.pth \
      checkpoints/R1q_femu_synthonly_30k_FINAL.pth

echo "" && echo "=== EXP2 STAGE 4: synth-only qhead 60k ===" && date
run_qhead R1q_femu_synthonly_60k checkpoints/tiny_femu_synthonly_60k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma
echo "=== EXP2 STAGE 4 DONE ===" && date
cp -f checkpoints/R1q_femu_synthonly_60k_best.pth \
      checkpoints/R1q_femu_synthonly_60k_FINAL.pth

# ============================================================
# EXP3 — RevEWMNorm span sweep
# ============================================================
# span=32 = existing fe+mu; we only need 64, 128, 256 here.
for SPAN in 64 128 256; do
    NAME="tiny_femu_span${SPAN}"
    echo "" && echo "=== EXP3 STAGE: span=${SPAN} backbone ===" && date
    run_train_backbone "$NAME" \
        --mix-ratio 0.5 --freq-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${NAME}_best_gap.pth" "checkpoints/${NAME}_FINAL.pth"
    echo "=== EXP3 STAGE: span=${SPAN} backbone DONE ===" && date

    QNAME="R1q_femu_span${SPAN}"
    echo "" && echo "=== EXP3 STAGE: span=${SPAN} qhead ===" && date
    run_qhead "$QNAME" "checkpoints/${NAME}_FINAL.pth" \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${QNAME}_best.pth" "checkpoints/${QNAME}_FINAL.pth"
    echo "=== EXP3 STAGE: span=${SPAN} qhead DONE ===" && date

    # Cheap screen: full GIFT-Eval — for span sweep we want the same
    # 97-config aggregate as the baseline so comparison is direct.
    # If wallclock pressures, swap to a shorter --max-configs.
    echo "" && echo "=== EXP3 STAGE: span=${SPAN} GIFT-Eval ===" && date
    run_eval_full "${QNAME}" "checkpoints/${NAME}_FINAL.pth" "checkpoints/${QNAME}_FINAL.pth" \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    echo "=== EXP3 STAGE: span=${SPAN} eval DONE ===" && date
done

# ============================================================
# EXP4 — Patch-stats feature
# ============================================================
echo "" && echo "=== EXP4 STAGE 1: patch-stats backbone (fe+mu, span=32, patch_stats=diff) ===" && date
run_train_backbone tiny_femu_pstats \
    --mix-ratio 0.5 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 32 \
    --patch-stats diff
cp -f checkpoints/tiny_femu_pstats_best_gap.pth checkpoints/tiny_femu_pstats_FINAL.pth
echo "=== EXP4 STAGE 1 DONE ===" && date

echo "" && echo "=== EXP4 STAGE 2: patch-stats qhead ===" && date
run_qhead R1q_femu_pstats checkpoints/tiny_femu_pstats_FINAL.pth \
    --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/R1q_femu_pstats_best.pth checkpoints/R1q_femu_pstats_FINAL.pth
echo "=== EXP4 STAGE 2 DONE ===" && date

echo "" && echo "=== EXP4 STAGE 3: patch-stats GIFT-Eval (full 97) ===" && date
run_eval_full R1q_femu_pstats \
    checkpoints/tiny_femu_pstats_FINAL.pth \
    checkpoints/R1q_femu_pstats_FINAL.pth \
    --rev-norm-kind ewma --rev-norm-span 32
echo "=== EXP4 STAGE 3 DONE ===" && date

echo "" && echo "=== ALL EXPERIMENTS COMPLETE ===" && date
ls -la results/ checkpoints/*_FINAL.pth | head -40
