#!/bin/bash
# Exp 4: cumulative contrastive-loss extension (assumes Exp 3 wins).
# Loss: cosine_similarity_batch_add_pos_htft_add_f_cross_negs (PR #181) —
# adds (h_t, f_t) same-channel positive AND f_t cross-batch/cross-channel
# negatives on top of the τ=0.20 baseline.
# Architecture: gru @ τ=0.20, 15k steps fresh from scratch.
#
# Speculative vast.ai 5090 spot run. Adapted from
# experiments/2026-05-08_exp_tau_sweep/scripts/run_tau_sweep_vast_arm5.sh
# (CUDA fixes for cu128 / driver 575 inherited verbatim).
#
# Usage (on the vast container):
#   nohup bash experiments/2026-05-09_exp_loss_extensions/scripts/run_exp4_fnegs_vast.sh \
#       > /workspace/app/run_exp4_fnegs.log 2>&1 &
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
LOSS="cosine_similarity_batch_add_pos_htft_add_f_cross_negs"
SAVE_DIR="checkpoints"
TAU="0.20"
NAME="exp4_fnegs_tau_0_20"
MAX_RETRIES=5
RETRY_SLEEP=60

mkdir -p "${SAVE_DIR}"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP exp4 fnegs (${SAVE_DIR}/${NAME}_FINAL.pth exists) ===" && date
    exit 0
fi

echo "=== EXP4 fnegs (gru, τ=${TAU}, 15k) → run_name=${NAME} ===" && date
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
        --encoder-type gru \
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
    echo "=== EXP4 FAILED after ${MAX_RETRIES} attempts (rc=${rc}) ===" && date
    exit "$rc"
fi

# Promote best_loss -> FINAL (model + optimizer) so scp-back from elisa
# sees both files with the expected names.
if [ ! -f "${SAVE_DIR}/${NAME}_best_loss.pth" ]; then
    echo "=== EXP4 ERROR — ${NAME}_best_loss.pth missing post-train ==="
    exit 1
fi
cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
if [ -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" ]; then
    cp -f "${SAVE_DIR}/${NAME}_best_loss_optimizer.pth" \
          "${SAVE_DIR}/${NAME}_FINAL_optimizer.pth"
fi

echo "" && echo "=== EXP4 DONE ===" && date
ls -la "${SAVE_DIR}/${NAME}_FINAL"* 2>&1
exit 0
