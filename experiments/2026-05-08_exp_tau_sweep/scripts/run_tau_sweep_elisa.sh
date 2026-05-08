#!/bin/bash
# τ sweep — elisa-local launcher.
#
# Same recipe as scripts/run_tau_sweep.sh (vast.ai container) but adapted for
# elisa: cwd = repo checkout, CUDA_VISIBLE_DEVICES inherited from the env
# (default: 1 — the second 4090, normally idle), save-dir = sync_tau_sweep/
# under the main checkout (per CLAUDE.md rule 4).
#
# Usage:
#   nohup bash experiments/2026-05-08_exp_tau_sweep/scripts/run_tau_sweep_elisa.sh \
#       > sync_tau_sweep/run.log 2>&1 &
#
# Override the GPU:
#   CUDA_VISIBLE_DEVICES=0 nohup bash experiments/.../run_tau_sweep_elisa.sh ...
#
# Idempotent: skips arms whose <run>_FINAL.pth already exists.

set -e
cd /home/jupyter/contrastive-forecasting

export PYTHONPATH=/home/jupyter/contrastive-forecasting
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"
SAVE_DIR="sync_tau_sweep/checkpoints"

mkdir -p "${SAVE_DIR}"

train_one() {
    local TAU="$1"
    local SAFE_TAU=${TAU//./_}
    local NAME="tau_sweep_${SAFE_TAU}"

    if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
        echo "=== SKIP τ=${TAU} (${SAVE_DIR}/${NAME}_FINAL.pth exists) ===" && date
        return 0
    fi

    echo "" && echo "=== ARM τ=${TAU} → run_name=${NAME} ===" && date

    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        --device cuda --total-steps 50000 --batch-size 256 \
        --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
        --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
        --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --mix-ratio 0.0 \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span 128 \
        --tau "${TAU}" \
        --loss-shape "${LOSS}"

    cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
    echo "=== ARM τ=${TAU} DONE — saved ${NAME}_FINAL.pth ===" && date
}

TAUS=(0.03 0.05 0.07 0.10 0.20)
for TAU in "${TAUS[@]}"; do
    train_one "${TAU}"
done

echo "" && echo "=== τ sweep ALL DONE ===" && date
exit 0
