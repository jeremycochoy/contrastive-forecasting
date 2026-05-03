#!/bin/bash
# Three-arm comparison of reversible normalisers on a dual-axis (freq +
# seasonality) embedding stack, evaluated on GIFT-Eval.
#
#   Arm A: RevIN
#   Arm B: RevEWMNorm span=512  (best on synth — 2026-04-28_exp_csb_pair_span512)
#   Arm C: RevEWMNorm span=128  (best ema_loss on real — 2026-04-27_exp_span_sweep_real)
#
# Shared knobs:
#   * 30k bb + 30k qhead, batch 24, lr 1e-4 (bb) / 3e-4 (qh)
#   * Loss: cosine_similarity_batch (won the pair A/B by ~5–13% MASE)
#   * mix_ratio 0.5  (50% bundle base_mixed_v1 + 50% on-the-fly synth)
#   * freq_emb_dim 3  +  seasonality_emb_dim 3
#   * mixup_p 0.3 (mixed across both label embeddings)
#   * Selector: _best_loss → FINAL.pth (gap saturates early)
#
# Eval: full GIFT-Eval (97 configs) with the official evaluator. The
# backbone reads `dataset.freq` and `get_seasonality(dataset.freq)` per
# task and tags itself with the matching (freq_id, seasonality_id) so
# the embeddings carry meaningful per-task labels at eval time.
set -e
cd /workspace/app
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_dualemb_3arm: starting ===" && date

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"
LOSS="cosine_similarity_batch"

run_backbone() {
    local NAME=$1; shift
    local NORM_KIND=$1; shift
    local NORM_SPAN=$1; shift
    local SPAN_FLAG=""
    if [ "$NORM_KIND" = "ewma" ]; then
        SPAN_FLAG="--rev-norm-span $NORM_SPAN"
    fi
    python3 -u experiments/freq-embedding/scripts/train.py \
        --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --save-every 2000 --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        --mix-ratio 0.5 \
        --freq-emb-dim 3 --seasonality-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
        --loss-shape "$LOSS"
    cp -f "checkpoints/${NAME}_best_loss.pth" "checkpoints/${NAME}_FINAL.pth"
}

run_qhead() {
    local QNAME=$1; shift
    local BB=$1; shift
    local NORM_KIND=$1; shift
    local NORM_SPAN=$1; shift
    local SPAN_FLAG=""
    if [ "$NORM_KIND" = "ewma" ]; then
        SPAN_FLAG="--rev-norm-span $NORM_SPAN"
    fi
    python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-every 1000 --save-dir checkpoints --run-name "$QNAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        --mix-ratio 0.5 --rev-norm-kind "$NORM_KIND" $SPAN_FLAG \
        --reconstruction forecaster
    cp -f "checkpoints/${QNAME}_best.pth" "checkpoints/${QNAME}_FINAL.pth"
}

run_gift_eval() {
    local OUT_DIR=$1; shift
    local BB=$1; shift
    local QH=$1; shift
    local NORM_KIND=$1; shift
    local NORM_SPAN=$1; shift
    local SPAN_FLAG=""
    if [ "$NORM_KIND" = "ewma" ]; then
        SPAN_FLAG="--rev-norm-span $NORM_SPAN"
    fi
    python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
        --backbone-path "$BB" --head-path "$QH" \
        --output-dir "$OUT_DIR" --strategy B4 --forecast-len 16 \
        --rev-norm-kind "$NORM_KIND" $SPAN_FLAG --device cuda
}

mkdir -p results

# ===== Arm A: RevIN =====
ABB="tiny_dualemb_revin"
AQH="R1q_dualemb_revin"
echo "" && echo "=== ARM A STAGE B: $ABB (RevIN) ===" && date
run_backbone "$ABB" "revin" 0
echo "=== ARM A STAGE B DONE ===" && date

echo "" && echo "=== ARM A STAGE H: $AQH ===" && date
run_qhead "$AQH" "checkpoints/${ABB}_FINAL.pth" "revin" 0
echo "=== ARM A STAGE H DONE ===" && date

echo "" && echo "=== ARM A STAGE E: gift_eval ===" && date
run_gift_eval "results/gift_eval_revin" \
    "checkpoints/${ABB}_FINAL.pth" "checkpoints/${AQH}_FINAL.pth" "revin" 0
echo "=== ARM A STAGE E DONE ===" && date

# ===== Arm B: RevEWMNorm span=512 =====
BBB="tiny_dualemb_ewma512"
BQH="R1q_dualemb_ewma512"
echo "" && echo "=== ARM B STAGE B: $BBB (ewma span=512) ===" && date
run_backbone "$BBB" "ewma" 512
echo "=== ARM B STAGE B DONE ===" && date

echo "" && echo "=== ARM B STAGE H: $BQH ===" && date
run_qhead "$BQH" "checkpoints/${BBB}_FINAL.pth" "ewma" 512
echo "=== ARM B STAGE H DONE ===" && date

echo "" && echo "=== ARM B STAGE E: gift_eval ===" && date
run_gift_eval "results/gift_eval_ewma512" \
    "checkpoints/${BBB}_FINAL.pth" "checkpoints/${BQH}_FINAL.pth" "ewma" 512
echo "=== ARM B STAGE E DONE ===" && date

# ===== Arm C: RevEWMNorm span=128 =====
CBB="tiny_dualemb_ewma128"
CQH="R1q_dualemb_ewma128"
echo "" && echo "=== ARM C STAGE B: $CBB (ewma span=128) ===" && date
run_backbone "$CBB" "ewma" 128
echo "=== ARM C STAGE B DONE ===" && date

echo "" && echo "=== ARM C STAGE H: $CQH ===" && date
run_qhead "$CQH" "checkpoints/${CBB}_FINAL.pth" "ewma" 128
echo "=== ARM C STAGE H DONE ===" && date

echo "" && echo "=== ARM C STAGE E: gift_eval ===" && date
run_gift_eval "results/gift_eval_ewma128" \
    "checkpoints/${CBB}_FINAL.pth" "checkpoints/${CQH}_FINAL.pth" "ewma" 128
echo "=== ARM C STAGE E DONE ===" && date

echo "" && echo "=== run_dualemb_3arm: ALL DONE ===" && date
echo ""
echo "GIFT-Eval summaries:"
for d in revin ewma512 ewma128; do
    if [ -f "results/gift_eval_${d}/summary.txt" ]; then
        echo "--- ${d} ---"
        head -20 "results/gift_eval_${d}/summary.txt"
    fi
done
