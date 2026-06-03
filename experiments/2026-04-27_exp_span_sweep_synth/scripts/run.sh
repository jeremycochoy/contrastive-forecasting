#!/bin/bash
# RevEWMNorm span sweep on synth-only data (adapted from original task #11
# to match the synth-only methodology we settled on).
#
# Arms:
#   span=32  -- already trained as the baseline (B1 fe+mu @ 30k synth30k)
#   span=64  -- new
#   span=128 -- new
#   span=256 -- new
# (Optional follow-up: span=512 if 256 strictly wins.)
#
# Each arm: 30k backbone (mix=1.0, freq_emb=3, mixup=0.3) + 30k qhead +
# synth_eval against the same held-out 1024-sample test set used by the
# baseline+pstats comparison (seed=99999999).
#
# Results append to results/synth_eval/all_results.csv so every arm sits
# in one table.
set -e
cd /workspace/app

exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_span_sweep_synth: starting ===" && date

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

run_qhead() {
    local NAME=$1; shift
    local BB=$1; shift
    python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-every 1000 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        "$@"
}

run_eval_synth() {
    local ARM=$1; shift; local BB=$1; shift; local HEAD=$1; shift
    python3 -u experiments/2026-04-27_freq-embedding/scripts/synth_eval.py \
        --backbone "$BB" --head "$HEAD" \
        --arm "$ARM" --n-samples 1024 --batch-size 64 \
        --out-csv results/synth_eval/all_results.csv \
        --device cuda \
        "$@"
}

for SPAN in 64 128 256; do
    BBNAME="tiny_femu_span${SPAN}_synth30k"
    QHNAME="R1q_femu_span${SPAN}_synth30k"

    echo "" && echo "=== SS STAGE B-span${SPAN}: backbone @ 30k (synth-only, span=${SPAN}) ===" && date
    run_train_backbone "$BBNAME" 30000 \
        --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${BBNAME}_best_gap.pth" "checkpoints/${BBNAME}_FINAL.pth"
    echo "=== SS STAGE B-span${SPAN} DONE ===" && date

    echo "" && echo "=== SS STAGE H-span${SPAN}: qhead ===" && date
    run_qhead "$QHNAME" "checkpoints/${BBNAME}_FINAL.pth" \
        --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span "$SPAN"
    cp -f "checkpoints/${QHNAME}_best.pth" "checkpoints/${QHNAME}_FINAL.pth"
    echo "=== SS STAGE H-span${SPAN} DONE ===" && date

    echo "" && echo "=== SS STAGE E-span${SPAN}: synth eval ===" && date
    run_eval_synth "fe+mu @ 30k span=${SPAN}" \
        "checkpoints/${BBNAME}_FINAL.pth" \
        "checkpoints/${QHNAME}_FINAL.pth" \
        --rev-norm-span "$SPAN"
    echo "=== SS STAGE E-span${SPAN} DONE ===" && date
done

echo ""
echo "=== run_span_sweep_synth: ALL DONE — full table: ==="
cat results/synth_eval/all_results.csv
date
