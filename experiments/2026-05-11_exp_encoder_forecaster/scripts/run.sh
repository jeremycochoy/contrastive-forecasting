#!/bin/bash
# encoder-forecaster v2 — dropkey=0.7 — τ=0.10, 50k+ steps, on elisa GPU 1
#
# Architecture: GRU patch embedding → 6× causal transformer (encoder)
#               → 6× causal transformer (forecaster). Encoder output is
#               the contrastive target (x_original captured AFTER the
#               encoder, BEFORE the forecaster). L2 normalisation in the
#               loss creates the encoder/forecaster asymmetry.
#
# Architecture/HP — IDENTICAL to tau_sweep_0_10 baseline plus:
#   --num-encoder-layers 6   (NEW: 6 causal layers before the forecaster)
#   --encoder-dropkey 0.7    (NEW: defends against position-counting
#                             shortcut diagnosed in the 2026-05-10 failed
#                             experiment — drops 70% of below-diagonal
#                             attention edges per encoder layer per step.
#                             v2/_pb_: mask is per-(B, head) independent
#                             — attempt-1 used a shared (T,T) mask and
#                             NaN'd at step 11700.)
#   --amp-dtype bf16         (NEW: torch.autocast(bfloat16) on forward+loss)
#   batch_size=256           (UNCHANGED — keeps negative count and τ valid)
#
# Run name: enc_fcst_dropkey07_pb_50k
# Save dir: /home/jupyter/contrastive-forecasting/checkpoints (main checkout)
#          per CLAUDE.md rule #4 — never put valuable artefacts in a worktree.

set -e
cd /home/jupyter/cf-encoder-forecaster-v2

export PYTHONPATH=/home/jupyter/cf-encoder-forecaster-v2
export CUDA_VISIBLE_DEVICES=1
# expandable_segments avoids OOM at B=256 with 12 transformer layers (smoke
# at fp32 baseline was fine; the encoder doubles activation memory and
# triggers fragmentation with the default allocator).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /home/jupyter/contrastive-forecasting/experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

NAME="enc_fcst_dropkey07_pb_50k"
SAVE_DIR="/home/jupyter/contrastive-forecasting/checkpoints"
LOG_DIR="/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

if [ -f "${SAVE_DIR}/${NAME}_FINAL.pth" ]; then
    echo "=== SKIP — ${SAVE_DIR}/${NAME}_FINAL.pth exists ==="
    exit 0
fi

TOTAL_STEPS="${TOTAL_STEPS:-50000}"

echo "=== START ${NAME} (steps=${TOTAL_STEPS}) ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps "${TOTAL_STEPS}" --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 5000 --save-dir "${SAVE_DIR}" --run-name "${NAME}" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --num-encoder-layers 6 --encoder-dropkey 0.7 --amp-dtype bf16 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.10 \
    --loss-shape "cosine_similarity_batch" \
    --encoder-type gru \
    2>&1 | tee -a "${LOG_DIR}/run_${NAME}.log"

cp -f "${SAVE_DIR}/${NAME}_best_loss.pth" "${SAVE_DIR}/${NAME}_FINAL.pth"
echo "=== DONE ${NAME} — saved ${NAME}_FINAL.pth ===" && date
