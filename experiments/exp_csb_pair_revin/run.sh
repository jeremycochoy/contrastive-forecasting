#!/bin/bash
# Clean A/B comparison of contrastive losses, RevIN normaliser variant.
# Mirror of exp_csb_pair_span512 but with --rev-norm-kind revin (no span).
#
#   Arm C: cosine_similarity_batch_no_time_neg, RevIN
#   Arm D: cosine_similarity_batch,             RevIN
#
# Both arms: 30k bb + 30k qhead, mix=1.0, freq_emb=3, mixup=0.3,
# single-shot (no --resume), `_best_loss → _FINAL.pth` for both.
# Same eval as every prior arm: 1024 held-out synth samples,
# `synth_eval.py`, seed=99999999.
#
# Combined with the EWMA pair this gives a 4-arm comparison
# (EWMA × RevIN) × (no_time_neg × cosine_similarity_batch).
set -e
cd /workspace/app
exec >> >(tee -a /workspace/app/run_all.log) 2>&1
echo "" && echo "=== run_csb_pair_revin: starting ===" && date

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
        --rev-norm-kind revin \
        --loss-shape "$LOSS"
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
        --mix-ratio 1.0 --rev-norm-kind revin
    cp -f "checkpoints/${QNAME}_best.pth" "checkpoints/${QNAME}_FINAL.pth"
}

run_eval() {
    local ARM=$1; shift; local BB=$1; shift; local QH=$1; shift
    python3 -u experiments/freq-embedding/scripts/synth_eval.py \
        --backbone "$BB" --head "$QH" \
        --arm "$ARM" --n-samples 1024 --batch-size 64 \
        --out-csv results/synth_eval/all_results.csv \
        --device cuda --rev-norm-kind revin
}

mkdir -p results/synth_eval

# ===== Arm C: RevIN + no_time_neg =====
CBB="tiny_pair_revin_ntn"
CQH="R1q_pair_revin_ntn"
echo "" && echo "=== ARM C STAGE B: $CBB (revin + no_time_neg) ===" && date
run_backbone "$CBB" "cosine_similarity_batch_no_time_neg"
echo "=== ARM C STAGE B DONE ===" && date

echo "" && echo "=== ARM C STAGE H: $CQH ===" && date
run_qhead "$CQH" "checkpoints/${CBB}_FINAL.pth"
echo "=== ARM C STAGE H DONE ===" && date

echo "" && echo "=== ARM C STAGE E: synth eval ===" && date
run_eval "pair revin ntn (clean, best_loss)" \
    "checkpoints/${CBB}_FINAL.pth" "checkpoints/${CQH}_FINAL.pth"
echo "=== ARM C STAGE E DONE ===" && date

# ===== Arm D: RevIN + cosine_similarity_batch =====
DBB="tiny_pair_revin_csb"
DQH="R1q_pair_revin_csb"
echo "" && echo "=== ARM D STAGE B: $DBB (revin + csb) ===" && date
run_backbone "$DBB" "cosine_similarity_batch"
echo "=== ARM D STAGE B DONE ===" && date

echo "" && echo "=== ARM D STAGE H: $DQH ===" && date
run_qhead "$DQH" "checkpoints/${DBB}_FINAL.pth"
echo "=== ARM D STAGE H DONE ===" && date

echo "" && echo "=== ARM D STAGE E: synth eval ===" && date
run_eval "pair revin csb (clean, best_loss)" \
    "checkpoints/${DBB}_FINAL.pth" "checkpoints/${DQH}_FINAL.pth"
echo "=== ARM D STAGE E DONE ===" && date

echo "" && echo "=== run_csb_pair_revin: ALL DONE ===" && date
tail -3 results/synth_eval/all_results.csv
