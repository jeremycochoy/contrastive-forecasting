#!/bin/bash
# τ sweep: 5 from-scratch fixed-τ trainings.
#
# Each arm trains backbone-beta's architecture/HP from scratch at a fixed
# τ for 50 000 steps — "just enough to acquire knowledge". The winner can
# be extended later if 50k turns out to be too short. Reference for the
# learnable-τ baseline is backbone-beta itself (167k, τ ≈ 0.072 final);
# it is NOT an arm here (apples-to-apples = fixed-τ vs fixed-τ).
#
# τ values: 0.03, 0.05, 0.07, 0.10, 0.20 (5 arms)
#
# Recipe (matches backbone-beta arch/HP exactly; only τ is swept):
#   T_RAW=4096, C=1, d_model=384, n_heads=6, num_layers=6
#   freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128
#   AdamW lr=1e-3 wd=0.1 β1=0.9 β2=0.98
#   mixup_p=0.3, mix_ratio=0.0, loss_shape=cosine_similarity_batch
#   batch_size=256, total_steps=50000, save_every=5000
#
# After each arm: cp <run>_best_loss.pth <run>_FINAL.pth
# Idempotent: any arm whose <run>_FINAL.pth already exists is skipped.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

train_one() {
    local TAU="$1"
    local SAFE_TAU=${TAU//./_}
    local NAME="tau_sweep_${SAFE_TAU}"

    if [ -f "checkpoints/${NAME}_FINAL.pth" ]; then
        echo "=== SKIP τ=${TAU} (checkpoints/${NAME}_FINAL.pth exists) ===" && date
        return 0
    fi

    echo "" && echo "=== ARM τ=${TAU} → run_name=${NAME} ===" && date

    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        --device cuda --total-steps 50000 --batch-size 256 \
        --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
        --save-every 5000 --save-dir checkpoints --run-name "${NAME}" \
        --hf-repo "${HF_REPO}" --hf-path "${HF_PATH}" \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --mix-ratio 0.0 \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span 128 \
        --tau "${TAU}" \
        --loss-shape "${LOSS}"

    cp -f "checkpoints/${NAME}_best_loss.pth" "checkpoints/${NAME}_FINAL.pth"
    echo "=== ARM τ=${TAU} DONE — saved ${NAME}_FINAL.pth ===" && date
}

TAUS=(0.03 0.05 0.07 0.10 0.20)
for TAU in "${TAUS[@]}"; do
    train_one "${TAU}"
done

echo "" && echo "=== τ sweep ALL DONE ===" && date
exit 0
