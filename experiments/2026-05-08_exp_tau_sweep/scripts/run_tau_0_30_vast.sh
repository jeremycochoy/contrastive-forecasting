#!/bin/bash
# τ sweep extension — vast.ai single-arm launcher (τ=0.30 only).
#
# Adds a τ=0.30 point to the existing τ-sweep (0.05/0.07/0.10/0.20). Recipe is
# identical to run_tau_sweep_vast_arm5.sh except --tau is 0.30 and the run-name
# is tau_sweep_0_30. Trains 15k steps on a vast.ai 5090 data-center on-demand
# instance.
#
# Robustness: HF API has been intermittently flaky, so the python invocation
# is wrapped in a retry loop (5 tries, 60 s sleep between).
#
# Idempotent: exits immediately if checkpoints/tau_sweep_0_30_FINAL.pth
# already exists.
#
# Usage (on the vast container):
#   nohup bash experiments/2026-05-08_exp_tau_sweep/scripts/run_tau_0_30_vast.sh \
#       > /workspace/app/run_tau_0_30.log 2>&1 &
#   disown

set -u
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Vast.ai 5090 image ships /usr/local/cuda-13.0/compat/libcuda.so.580.65 which
# refuses to forward-load on the host driver 575.51 (error 804). Pointing
# LD_LIBRARY_PATH at the host driver libs bypasses the compat shim.
# Additionally the image bundles cuDNN 9.12 but the cu128 torch wheel was
# compiled against 9.19, so we prepend the wheel-bundled libs (under
# nvidia/*/lib + torch/lib) so the bundled cuDNN/CUDA runtime wins.
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
TAU="0.30"
NAME="tau_sweep_0_30"
MAX_RETRIES=5
RETRY_SLEEP=60

mkdir -p "${SAVE_DIR}"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP τ=${TAU} (${SAVE_DIR}/${NAME}_FINAL.pth exists) ===" && date
    exit 0
fi

echo "=== ARM τ=${TAU} → run_name=${NAME} ===" && date
echo "=== host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi -L 2>&1 | head -3 || true

# If a periodic save exists from a prior failed attempt, resume from the
# largest <run>_<step>k.pth that has a companion _optimizer.pth.
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

    RESUME_ARGS=()
    RESUME_PATH=$(detect_resume)
    if [ -n "$RESUME_PATH" ]; then
        echo "    resuming from ${RESUME_PATH}"
        RESUME_ARGS=(--resume "$RESUME_PATH")
    fi

    set +e
    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        --device cuda --total-steps 15000 --batch-size 256 \
        --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
        --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
        --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --mix-ratio 0.0 \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span 128 \
        --tau "${TAU}" \
        --loss-shape "${LOSS}" \
        "${RESUME_ARGS[@]}"
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
    echo "=== ARM τ=${TAU} FAILED after ${MAX_RETRIES} attempts (rc=${rc}) ===" && date
    exit "$rc"
fi

# Promote best_loss -> FINAL (model + optimizer) so scp-back from elisa
# sees both files with the expected names.
if [ ! -f "${SAVE_DIR}/${NAME}_best_loss.pth" ]; then
    echo "=== ARM τ=${TAU} ERROR — ${NAME}_best_loss.pth missing post-train ==="
    exit 1
fi
cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
if [ -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" ]; then
    cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" \
          "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
fi

echo "" && echo "=== ARM τ=0.30 DONE ===" && date
ls -la "${SAVE_DIR}/${NAME}_FINAL"* 2>&1
exit 0
