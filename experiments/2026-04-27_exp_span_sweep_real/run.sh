#!/bin/bash
# RevEWMNorm span sweep on PURE REAL data (mix=0.0, base-bundles).
# Question: which EWMA span best matches the real-world non-stationarity
# in our training distribution? mix=0 isolates the answer from synthetic
# confounds.
#
# Backbones only (option A from the discussion). The contrastive loss/gap
# is a per-step real-data metric — same kind the original ARMA span sweep
# used to settle on span=32 (experiments/2026-04-12_revnorm-span-search/report.md).
# No qheads, no GIFT-Eval — just compare training trajectories.
#
# Spans: {32, 64, 128, 256}. 4 backbones × 30k steps. ~$2.30 estimate.
#
# Optional follow-up (only if compute remains): same sweep on mix=1.0
# synth-only — see run_span_sweep_synth.sh for that variant.
set -e
cd /workspace/app

exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_span_sweep_real: starting ===" && date

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"

run_train_backbone() {
    local NAME=$1; shift; local STEPS=$1; shift
    python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
        --device cuda --total-steps "$STEPS" --batch-size 24 --lr 1e-4 \
        --save-every 2000 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        "$@"
}

for SPAN in 32 64 128 256; do
    NAME="tiny_femu_real_span${SPAN}"

    echo "" && echo "=== SS-real STAGE: span=${SPAN} backbone (mix=0.0) ===" && date
    run_train_backbone "$NAME" 30000 \
        --mix-ratio 0.0 --freq-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${NAME}_best_gap.pth" "checkpoints/${NAME}_FINAL.pth"
    echo "=== SS-real STAGE: span=${SPAN} DONE ===" && date
done

echo ""
echo "=== run_span_sweep_real: ALL DONE ==="
echo "Final summary (last loss / best gap per span):"
for SPAN in 32 64 128 256; do
    CSV="checkpoints/tiny_femu_real_span${SPAN}_losses.csv"
    if [[ -s "$CSV" ]]; then
        last=$(tail -1 "$CSV")
        echo "  span=${SPAN}  last row: $last"
    fi
done
date
