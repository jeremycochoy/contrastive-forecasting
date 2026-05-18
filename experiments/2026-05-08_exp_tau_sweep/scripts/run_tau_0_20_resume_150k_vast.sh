#!/bin/bash
# τ=0.20 resume — 50k → 150k vast.ai single-arm launcher.
#
# Continue τ=0.20 from its 50k FINAL bundle to 150,000 steps total on a
# fresh vast.ai DC GPU on-demand instance. Same setup as τ=0.10 sister
# launcher, only TAU + NAME + RESUME paths differ.
#
# Resume bundle pushed pre-launch:
#   checkpoints/tau_sweep_0_20_resume_150k.pth           (= local _r2_FINAL.pth)
#   checkpoints/tau_sweep_0_20_resume_150k_optimizer.pth (= local _r2_FINAL_optimizer.pth)
#
# Usage (on the vast container):
#   nohup bash experiments/2026-05-08_exp_tau_sweep/scripts/run_tau_0_20_resume_150k_vast.sh \
#       > /workspace/app/run_tau_0_20_150k.log 2>&1 &
#   disown

set -u
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

TORCH_LIB_DIR=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
NVIDIA_PKG=/usr/local/lib/python3.12/dist-packages/nvidia
NVIDIA_DIRS=""
if [ -d "$NVIDIA_PKG" ]; then
    for d in "$NVIDIA_PKG"/*/lib; do
        [ -d "$d" ] && NVIDIA_DIRS="${NVIDIA_DIRS:+${NVIDIA_DIRS}:}$d"
    done
fi
export LD_LIBRARY_PATH="${NVIDIA_DIRS}:${TORCH_LIB_DIR}:/usr/lib/x86_64-linux-gnu"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"
SAVE_DIR="checkpoints"
TAU="0.20"
NAME="tau_sweep_0_20_150k"
RESUME="checkpoints/tau_sweep_0_20_resume_150k.pth"
MAX_RETRIES=5
RETRY_SLEEP=60

mkdir -p "${SAVE_DIR}"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP τ=${TAU} 150k (${SAVE_DIR}/${NAME}_FINAL.pth exists) ===" && date
    exit 0
fi

if [ ! -f "${RESUME}" ]; then
    echo "FATAL: resume file ${RESUME} missing" && date
    exit 1
fi

echo "=== ARM τ=${TAU} RESUME 50k → 150k → run_name=${NAME} ===" && date
echo "=== host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi -L 2>&1 | head -3 || true

detect_resume() {
    local latest=""
    local latest_step=-1
    for f in "${SAVE_DIR}/${NAME}"_*k.pth; do
        [ -e "$f" ] || continue
        case "$f" in
            *_optimizer.pth) continue ;;
        esac
        local opt="${f%.pth}_optimizer.pth"
        [ -f "$opt" ] || continue
        local base
        base=$(basename "$f" .pth)
        local step="${base##*_}"
        step="${step%k}"
        case "$step" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ "$step" -gt "$latest_step" ]; then
            latest_step="$step"
            latest="$f"
        fi
    done
    echo "$latest"
}

attempt=1
rc=1
while [ "$attempt" -le "$MAX_RETRIES" ]; do
    echo "" && echo "--- attempt ${attempt}/${MAX_RETRIES} ---" && date

    RESUME_PATH=$(detect_resume)
    if [ -z "$RESUME_PATH" ]; then
        if [ -f "${SAVE_DIR}/${NAME}_best_loss.pth" ] && \
           [ -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" ]; then
            RESUME_PATH="${SAVE_DIR}/${NAME}_best_loss.pth"
        else
            RESUME_PATH="${RESUME}"
        fi
    fi
    echo "    resuming from ${RESUME_PATH}"

    set +e
    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        --device cuda:0 --total-steps 150000 --batch-size 256 \
        --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
        --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
        --resume "${RESUME_PATH}" \
        --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --mix-ratio 0.0 \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span 128 \
        --encoder-type gru \
        --tau "${TAU}" \
        --loss-shape "${LOSS}"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        break
    fi
    echo "[retry] attempt ${attempt} failed with rc=${rc}; sleeping ${RETRY_SLEEP}s" && date
    sleep "$RETRY_SLEEP"
    attempt=$((attempt + 1))
done

if [ "$rc" -ne 0 ]; then
    echo "=== ARM τ=${TAU} 150k FAILED after ${MAX_RETRIES} attempts (rc=${rc}) ===" && date
    exit "$rc"
fi

PROMOTE_NAME="${NAME}"
if [ -f "${SAVE_DIR}/${NAME}_r2_best_loss.pth" ]; then
    PROMOTE_NAME="${NAME}_r2"
fi
if [ ! -f "${SAVE_DIR}/${PROMOTE_NAME}_best_loss.pth" ]; then
    echo "=== ARM τ=${TAU} 150k ERROR — ${PROMOTE_NAME}_best_loss.pth missing post-train ==="
    exit 1
fi
cp -f "${SAVE_DIR}/${PROMOTE_NAME}_best_loss.pth" "${SAVE_DIR}/${PROMOTE_NAME}_FINAL.pth"
if [ -f "${SAVE_DIR}/${PROMOTE_NAME}_best_loss_optimizer.pth" ]; then
    cp -f "${SAVE_DIR}/${PROMOTE_NAME}_best_loss_optimizer.pth" \
          "${SAVE_DIR}/${PROMOTE_NAME}_FINAL_optimizer.pth"
fi

echo "" && echo "=== ARM τ=0.20 150k DONE ===" && date
ls -la "${SAVE_DIR}/${NAME}"* 2>&1 | head -20
exit 0
