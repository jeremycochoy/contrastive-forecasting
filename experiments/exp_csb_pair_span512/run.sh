#!/bin/bash
# Clean A/B comparison of contrastive losses on the span=512 best arm:
#   Arm A: cosine_similarity_batch_no_time_neg (the previous default — was
#          the lost baseline)
#   Arm B: cosine_similarity_batch (paper-matching, includes within-time
#          and cross-time negatives — what exp_csb_synth used, but in a
#          multi-resume run; this redoes it clean)
#
# Both arms: 30k bb + 30k qhead, mix=1.0, freq_emb=3, mixup=0.3,
# RevEWMNorm span=512, single-shot (no --resume), _best_loss → FINAL.pth.
# The default save-every keeps periodic snapshots and _best_gap.pth so
# we can re-eval with a different selector later if we want.
#
# Eval: same held-out 1024-sample synth set (seed=99999999), same
# synth_eval.py protocol as every prior arm in
# experiments/_aggregate/results/synth_eval.csv.
set -e
cd /workspace/app
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_csb_pair_span512: starting ===" && date

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

HF_REPO="jeremycochoy/contrastive-training-base-bundles"
HF_PATH="base_mixed_v1"

run_backbone() {
    local NAME=$1; shift
    local LOSS=$1; shift
    python3 -u experiments/freq-embedding/scripts/train.py \
        --device cuda --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --save-every 2000 --save-dir checkpoints --run-name "$NAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" \
        --mix-ratio 1.0 --freq-emb-dim 3 --mixup-p 0.3 \
        --rev-norm-kind ewma --rev-norm-span 512 \
        --loss-shape "$LOSS"
    # _best_loss is the FINAL selector (gap saturates early on synth).
    # _best_gap.pth and periodic *_Nk.pth are preserved on disk.
    cp -f "checkpoints/${NAME}_best_loss.pth" "checkpoints/${NAME}_FINAL.pth"
}

run_qhead() {
    local QNAME=$1; shift
    local BB=$1; shift
    python3 -u experiments/gift-eval/scripts/train_forecasting_head.py \
        --backbone-path "$BB" --forecast-len 16 --quantile-head \
        --total-steps 30000 --batch-size 24 --lr 3e-4 \
        --save-every 1000 --save-dir checkpoints --run-name "$QNAME" \
        --hf-repo "$HF_REPO" --hf-path "$HF_PATH" --device cuda \
        --mix-ratio 1.0 --rev-norm-kind ewma --rev-norm-span 512
    cp -f "checkpoints/${QNAME}_best.pth" "checkpoints/${QNAME}_FINAL.pth"
}

run_eval() {
    local ARM=$1; shift; local BB=$1; shift; local QH=$1; shift
    python3 -u experiments/freq-embedding/scripts/synth_eval.py \
        --backbone "$BB" --head "$QH" \
        --arm "$ARM" --n-samples 1024 --batch-size 64 \
        --out-csv results/synth_eval/all_results.csv \
        --device cuda --rev-norm-span 512
}

mkdir -p results/synth_eval

# ===== Arm A: no_time_neg (the lost baseline, redone clean) =====
ABB="tiny_pair_span512_ntn"
AQH="R1q_pair_span512_ntn"
echo "" && echo "=== ARM A STAGE B: $ABB (no_time_neg) ===" && date
run_backbone "$ABB" "cosine_similarity_batch_no_time_neg"
echo "=== ARM A STAGE B DONE ===" && date

echo "" && echo "=== ARM A STAGE H: $AQH ===" && date
run_qhead "$AQH" "checkpoints/${ABB}_FINAL.pth"
echo "=== ARM A STAGE H DONE ===" && date

echo "" && echo "=== ARM A STAGE E: synth eval ===" && date
run_eval "pair span=512 ntn (clean, best_loss)" \
    "checkpoints/${ABB}_FINAL.pth" "checkpoints/${AQH}_FINAL.pth"
echo "=== ARM A STAGE E DONE ===" && date

# ===== Arm B: cosine_similarity_batch (paper loss, redone clean) =====
BBB="tiny_pair_span512_csb"
BQH="R1q_pair_span512_csb"
echo "" && echo "=== ARM B STAGE B: $BBB (csb) ===" && date
run_backbone "$BBB" "cosine_similarity_batch"
echo "=== ARM B STAGE B DONE ===" && date

echo "" && echo "=== ARM B STAGE H: $BQH ===" && date
run_qhead "$BQH" "checkpoints/${BBB}_FINAL.pth"
echo "=== ARM B STAGE H DONE ===" && date

echo "" && echo "=== ARM B STAGE E: synth eval ===" && date
run_eval "pair span=512 csb (clean, best_loss)" \
    "checkpoints/${BBB}_FINAL.pth" "checkpoints/${BQH}_FINAL.pth"
echo "=== ARM B STAGE E DONE ===" && date

echo "" && echo "=== run_csb_pair_span512: ALL DONE ===" && date
tail -3 results/synth_eval/all_results.csv
