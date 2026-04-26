#!/bin/bash
# Synth-only patch-stats experiment.
#
# Goal (per-user clarification): see whether the patch-stats architectural
# change makes a measurable difference on the model's *own training
# distribution* (clean periodic synth), where the freq embedding has perfect
# information so 30k steps may actually be enough for the backbone to learn
# the cyclical pattern. Compare 30k vs 60k to disentangle "needs more
# training" from "architecture limit".
#
# Arms:
#   1. fe+mu  baseline @ 30k bb + 30k qhead   (mix=1.0, no patch-stats)
#   2. fe+mu  baseline @ 60k bb + 30k qhead
#   3. fe+mu+pstats   @ 30k bb + 30k qhead   (mix=1.0, patch_stats=diff)
#   4. fe+mu+pstats   @ 60k bb + 30k qhead
#
# Eval:
#   Held-out synthetic test set (1024 samples × 4 channels) from
#   src.synthetic_periodic.generate_periodic_batch with seed=99999999
#   (never used during training).
#
# Reports per-arm GM-MASE / GM-WQL plus skill scores vs Seasonal-Naive
# (uses the known period — essentially the upper bound on this data).
set -e
cd /workspace/app

exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_synth_only: starting ===" && date

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"   # only used as a placeholder for mix=1.0; HF rows aren't fetched.

run_train_backbone() {
    # $1=run_name, $2=total_steps, $3..=extra args
    local NAME=$1; shift; local STEPS=$1; shift
    python3 -u experiments/freq-embedding/scripts/train.py \
        --device cuda --total-steps "$STEPS" --batch-size 24 --lr 1e-4 \
        --save-every 2000 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        "$@"
}

run_qhead() {
    # $1=run_name, $2=backbone path, $3..=extra args
    local NAME=$1; shift
    local BB=$1; shift
    python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-every 1000 \
        --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        "$@"
}

run_eval_synth() {
    # $1=arm_label, $2=backbone, $3=head
    local ARM=$1; shift; local BB=$1; shift; local HEAD=$1; shift
    python3 -u experiments/freq-embedding/scripts/synth_eval.py \
        --backbone "$BB" --head "$HEAD" \
        --arm "$ARM" --n-samples 1024 --batch-size 64 \
        --out-csv results/synth_eval/all_results.csv \
        --device cuda
}

# ============================================================
# BACKBONE 1: fe+mu baseline @ 30k (synth-only)
# ============================================================
echo "" && echo "=== SO STAGE B1: fe+mu @ 30k (mix=1.0, no pstats) ===" && date
run_train_backbone tiny_femu_synth30k 30000 \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/tiny_femu_synth30k_best_gap.pth checkpoints/tiny_femu_synth30k_FINAL.pth
echo "=== SO STAGE B1 DONE ===" && date

# ============================================================
# BACKBONE 2: fe+mu baseline @ 60k (synth-only)
# ============================================================
echo "" && echo "=== SO STAGE B2: fe+mu @ 60k (mix=1.0, no pstats) ===" && date
run_train_backbone tiny_femu_synth60k 60000 \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/tiny_femu_synth60k_best_gap.pth checkpoints/tiny_femu_synth60k_FINAL.pth
echo "=== SO STAGE B2 DONE ===" && date

# ============================================================
# BACKBONE 3: fe+mu+pstats @ 30k (synth-only)
# ============================================================
echo "" && echo "=== SO STAGE B3: fe+mu+pstats @ 30k (mix=1.0, patch_stats=diff) ===" && date
run_train_backbone tiny_femu_pstats_synth30k 30000 \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 32 \
    --patch-stats diff
cp -f checkpoints/tiny_femu_pstats_synth30k_best_gap.pth \
      checkpoints/tiny_femu_pstats_synth30k_FINAL.pth
echo "=== SO STAGE B3 DONE ===" && date

# ============================================================
# BACKBONE 4: fe+mu+pstats @ 60k (synth-only)
# ============================================================
echo "" && echo "=== SO STAGE B4: fe+mu+pstats @ 60k (mix=1.0, patch_stats=diff) ===" && date
run_train_backbone tiny_femu_pstats_synth60k 60000 \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 32 \
    --patch-stats diff
cp -f checkpoints/tiny_femu_pstats_synth60k_best_gap.pth \
      checkpoints/tiny_femu_pstats_synth60k_FINAL.pth
echo "=== SO STAGE B4 DONE ===" && date

# ============================================================
# QHEADs (synth-only, 30k each)
# ============================================================
echo "" && echo "=== SO STAGE H1: qhead on baseline 30k ===" && date
run_qhead R1q_femu_synth30k checkpoints/tiny_femu_synth30k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/R1q_femu_synth30k_best.pth checkpoints/R1q_femu_synth30k_FINAL.pth
echo "=== SO STAGE H1 DONE ===" && date

echo "" && echo "=== SO STAGE H2: qhead on baseline 60k ===" && date
run_qhead R1q_femu_synth60k checkpoints/tiny_femu_synth60k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/R1q_femu_synth60k_best.pth checkpoints/R1q_femu_synth60k_FINAL.pth
echo "=== SO STAGE H2 DONE ===" && date

echo "" && echo "=== SO STAGE H3: qhead on pstats 30k ===" && date
run_qhead R1q_femu_pstats_synth30k checkpoints/tiny_femu_pstats_synth30k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/R1q_femu_pstats_synth30k_best.pth checkpoints/R1q_femu_pstats_synth30k_FINAL.pth
echo "=== SO STAGE H3 DONE ===" && date

echo "" && echo "=== SO STAGE H4: qhead on pstats 60k ===" && date
run_qhead R1q_femu_pstats_synth60k checkpoints/tiny_femu_pstats_synth60k_FINAL.pth \
    --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 32
cp -f checkpoints/R1q_femu_pstats_synth60k_best.pth checkpoints/R1q_femu_pstats_synth60k_FINAL.pth
echo "=== SO STAGE H4 DONE ===" && date

# ============================================================
# EVAL: synth-only test set, all 4 arms
# ============================================================
mkdir -p results/synth_eval
# Reset the CSV so we don't append to a stale file.
rm -f results/synth_eval/all_results.csv

echo "" && echo "=== SO STAGE E1: eval baseline 30k ===" && date
run_eval_synth "fe+mu @ 30k" \
    checkpoints/tiny_femu_synth30k_FINAL.pth \
    checkpoints/R1q_femu_synth30k_FINAL.pth
echo "=== SO STAGE E1 DONE ===" && date

echo "" && echo "=== SO STAGE E2: eval baseline 60k ===" && date
run_eval_synth "fe+mu @ 60k" \
    checkpoints/tiny_femu_synth60k_FINAL.pth \
    checkpoints/R1q_femu_synth60k_FINAL.pth
echo "=== SO STAGE E2 DONE ===" && date

echo "" && echo "=== SO STAGE E3: eval pstats 30k ===" && date
run_eval_synth "fe+mu+pstats @ 30k" \
    checkpoints/tiny_femu_pstats_synth30k_FINAL.pth \
    checkpoints/R1q_femu_pstats_synth30k_FINAL.pth
echo "=== SO STAGE E3 DONE ===" && date

echo "" && echo "=== SO STAGE E4: eval pstats 60k ===" && date
run_eval_synth "fe+mu+pstats @ 60k" \
    checkpoints/tiny_femu_pstats_synth60k_FINAL.pth \
    checkpoints/R1q_femu_pstats_synth60k_FINAL.pth
echo "=== SO STAGE E4 DONE ===" && date

echo ""
echo "=== run_synth_only: ALL DONE — final table: ==="
cat results/synth_eval/all_results.csv
date
