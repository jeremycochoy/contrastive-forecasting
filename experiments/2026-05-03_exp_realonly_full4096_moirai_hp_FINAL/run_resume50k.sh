#!/bin/bash
# Resume FRESH run from step 50k checkpoint to test whether the deterministic
# resume code (PR #110: hf_rows_consumed + RNG cast fix) eliminates the
# +52% loss-std jump observed in v1/v2 of #10.
#
# IMPORTANT: writes to a NEW prefix `tiny_full4096_moirai_hp_FRESH_RESUME50k`
# so the existing FRESH 50k periodic + best_loss + best_gap checkpoints are
# NOT overwritten.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/gift-pretrain-full-4096"
HF_PATH="small_v1"
LOSS="cosine_similarity_batch"

# NEW distinct run names (do NOT clobber the FRESH 50k+ checkpoints)
BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
QH="R1q_full4096_moirai_hp_FRESH_RESUME50k"

# Resume from the 50k periodic save of the FRESH run
RESUME_BB="checkpoints/tiny_full4096_moirai_hp_FRESH_50k.pth"

if [ ! -f "${RESUME_BB}" ]; then
    echo "ERROR: resume backbone not found at ${RESUME_BB}" >&2
    exit 1
fi
if [ ! -f "${RESUME_BB%.pth}_optimizer.pth" ]; then
    echo "ERROR: resume optimizer not found at ${RESUME_BB%.pth}_optimizer.pth" >&2
    echo "       Without it, resume loses step counter, RNG, AdamW momentum, hf_rows_consumed." >&2
    exit 1
fi

echo "" && echo "=== run_full4096_moirai_hp_FRESH_RESUME50k: starting ===" && date
echo "" && echo "=== STAGE B: $BB (resume from FRESH 50k → 1 full epoch, MOIRAI HP) ===" && date

python3 -u experiments/freq-embedding/scripts/train.py \
    --resume "$RESUME_BB" \
    --device cuda --total-steps 167000 --batch-size 256 \
    --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --save-every 10000 --save-dir checkpoints --run-name "$BB" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --tau 0.07 --learnable-tau \
    --loss-shape "$LOSS"

cp -f "checkpoints/${BB}_best_loss.pth" "checkpoints/${BB}_FINAL.pth"
echo "=== STAGE B DONE ===" && date

# Backbone-only run; qhead+eval gated on remaining credit + user direction.
echo "=== run_full4096_moirai_hp_FRESH_RESUME50k: backbone-only completion ===" && date
echo "" && echo "=== run_full4096_moirai_hp_FRESH_RESUME50k: ALL DONE ===" && date
exit 0
