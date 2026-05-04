#!/bin/bash
# Recovery launcher for τ=0.07 arm of #27 after the original machine3
# (35970433) was host-stopped at backbone step 2k on 2026-05-01 ~19:37.
# We push the locally-synced 2k checkpoint pair to /workspace/app/resume_source/
# (separate dir so safe_run_name doesn't auto-branch the run name) and resume
# Stage B from there; Stages H and E run as normal.
set -e
cd /workspace/app

ARM="007"
TAU="0.07"
LOG="/workspace/app/run_tau${ARM}.log"
exec >> >(tee -a "$LOG") 2>&1
echo "" && echo "=== run_tau007_resume: tau=${TAU} (arm=${ARM}) RESUMING ===" && date

# ---------------------------------------------------------------------------
# Preflight gate — refuse to launch if the resume-bundle is incomplete.
# See docs/SYNC_PROTOCOL_REVIEW.md §3.1 (the four-file rule). Each
# missing file produces a ✗ line; we exit non-zero with a hint pointing
# at scripts/push_resume_bundle.sh in the local checkout.
#
# We pre-pick the resume .pth the same way the body does below
# (resume_source/${BB}_3600.pth or fall back to _2k.pth). Then we
# require its companion _optimizer.pth and the historical losses CSV
# (which the trainer's CSVLogger opens with mode="a"). Without the CSV,
# the trainer creates a fresh header-only file; the local sync_loop
# pulls it and overwrites the long good copy on its next tick.
# ---------------------------------------------------------------------------
_BB_PRE="tiny_realonly_4096_smaller_tau${ARM}"
if [ -f "/workspace/app/resume_source/${_BB_PRE}_3600.pth" ]; then
    _RESUME_BB_PRE="/workspace/app/resume_source/${_BB_PRE}_3600.pth"
else
    _RESUME_BB_PRE="/workspace/app/resume_source/${_BB_PRE}_2k.pth"
fi
_PREFLIGHT_MISSING=()
[ -f "$_RESUME_BB_PRE" ] || _PREFLIGHT_MISSING+=("backbone .pth: $_RESUME_BB_PRE")
[ -f "${_RESUME_BB_PRE%.pth}_optimizer.pth" ] || _PREFLIGHT_MISSING+=("backbone optimizer: ${_RESUME_BB_PRE%.pth}_optimizer.pth")
[ -f "/workspace/app/checkpoints/${_BB_PRE}_losses.csv" ] || _PREFLIGHT_MISSING+=("backbone losses CSV: /workspace/app/checkpoints/${_BB_PRE}_losses.csv")
# run log: this script is a *resume*, not a fresh start, so the log
# must already exist (otherwise `tee -a` creates a new short log and
# the local sync_loop overwrites the long good copy on its next tick).
[ -f "$LOG" ] || _PREFLIGHT_MISSING+=("run log (resume must extend, not reset): $LOG")
if [ "${#_PREFLIGHT_MISSING[@]}" -gt 0 ]; then
    echo "" >&2
    echo "ERROR: resume-bundle preflight failed. Missing files on this remote:" >&2
    for m in "${_PREFLIGHT_MISSING[@]}"; do echo "  ✗ $m" >&2; done
    echo "" >&2
    echo "Push them with scripts/push_resume_bundle.sh from the local checkout, then re-run." >&2
    echo "  e.g.:" >&2
    echo "    scripts/push_resume_bundle.sh \\" >&2
    echo "      <local_sync_arm_dir> <user>@<host> <ssh_port> $_BB_PRE R1q_realonly_4096_smaller_tau${ARM} tau${ARM}" >&2
    exit 2
fi
unset _PREFLIGHT_MISSING _BB_PRE _RESUME_BB_PRE

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
mkdir -p checkpoints

# Resume source (pre-pushed from local sync, plus more recent best_loss
# moved here after credit-restore). Kept in a separate directory so
# safe_run_name() doesn't see existing run-name checkpoints and branch us
# to "..._r2"; new checkpoints land cleanly in checkpoints/<run_name>_*.pth.
# Prefer 3600 (latest) if present, else fall back to 2k.
if [ -f "/workspace/app/resume_source/${BB}_3600.pth" ]; then
    RESUME_BB="/workspace/app/resume_source/${BB}_3600.pth"
else
    RESUME_BB="/workspace/app/resume_source/${BB}_2k.pth"
fi
if [ ! -f "$RESUME_BB" ]; then
    echo "ERROR: resume checkpoint missing at $RESUME_BB. Push it before running this script." >&2
    exit 2
fi

echo "" && echo "=== ARM ${ARM} STAGE B (RESUME from 2k): $BB (smaller, EWMA-128, tau=${TAU}, bs=96) ===" && date
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
    --loss-shape "$LOSS" \
    --resume "$RESUME_BB"
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

echo "" && echo "=== run_tau007_resume: tau=${TAU} (arm=${ARM}) ALL DONE ===" && date
echo ""
if [ -f "$RES_DIR/summary.txt" ]; then
    head -30 "$RES_DIR/summary.txt"
fi
