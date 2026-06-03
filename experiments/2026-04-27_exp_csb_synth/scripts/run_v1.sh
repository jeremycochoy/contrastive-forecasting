#!/bin/bash
# FOLLOWUP-1: re-introduce the within-series, within-channel time negative
# h[b, t-1, c] <> h[b, t, c] in the contrastive loss. Trained on the
# best-known arm config so the comparison is single-axis vs the existing
# fe+mu @ 30k span=512 synth-only baseline.
#
# Stages:
#   1. Backbone: fe+mu, mix=1.0 synth, 30k steps, span=512,
#      loss_shape='cosine_similarity_batch_with_within_time_neg'
#   2. Quantile head: 30k synth-only, on the new backbone
#   3. Synth eval: 1024 held-out samples (same seed/protocol as the
#      span sweep so the row is directly comparable)
set -e
cd /workspace/app
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_within_time_neg: starting ===" && date
export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"

NAME="tiny_femu_span512_synth30k_wtn"
QNAME="R1q_femu_span512_synth30k_wtn"

# 1. Backbone
echo "" && echo "=== WTN STAGE B: fe+mu @ 30k span=512 synth + within-time-neg ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/train.py \
    --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
    --save-every 2000 --save-dir checkpoints --run-name "$NAME" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
    --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
    --rev-norm-kind ewma --rev-norm-span 512 \
    --loss-shape cosine_similarity_batch_with_within_time_neg
cp -f checkpoints/${NAME}_30k.pth checkpoints/${NAME}_FINAL.pth
echo "=== WTN STAGE B DONE ===" && date

# 2. Qhead
echo "" && echo "=== WTN STAGE H: qhead 30k synth ===" && date
python3 -u experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py \
    --backbone-path "checkpoints/${NAME}_FINAL.pth" --forecast-len 16 --quantile-head \
    --total-steps 30000 --batch-size 24 --lr 3e-4 \
    --save-every 1000 --save-dir checkpoints --run-name "$QNAME" \
    --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
    --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 512
cp -f checkpoints/${QNAME}_best.pth checkpoints/${QNAME}_FINAL.pth
echo "=== WTN STAGE H DONE ===" && date

# 3. Synth eval — append to the existing CSV so the row sits next to
#    the span=512 baseline.
mkdir -p results/synth_eval
echo "" && echo "=== WTN STAGE E: synth eval ===" && date
python3 -u experiments/2026-04-27_freq-embedding/scripts/synth_eval.py \
    --backbone "checkpoints/${NAME}_FINAL.pth" --head "checkpoints/${QNAME}_FINAL.pth" \
    --arm "fe+mu @ 30k span=512 +within_time_neg" \
    --n-samples 1024 --batch-size 64 \
    --out-csv results/synth_eval/all_results.csv \
    --device cuda --rev-norm-span 512
echo "=== WTN STAGE E DONE ===" && date

echo "" && echo "=== run_within_time_neg: ALL DONE ===" && date
echo "Eval table tail:"
tail -5 results/synth_eval/all_results.csv
