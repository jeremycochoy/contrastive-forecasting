#!/bin/bash
# Generic chain script: train 2L qhead on the _40k.pth periodic backbone
# checkpoint of a given arm, then triage. Apples-to-apples vs v11c (whose
# FINAL = best_loss snapshot at step ~40.6k).
#
# Usage:  CUDA_VISIBLE_DEVICES=N bash post_qhead_chain_at40k.sh ARM
# where ARM ∈ {v13, v14, v15, v16, v17, v11c_at50k}. v11c "at40k" is skipped
# because its FINAL is already the ~40.6k best_loss snapshot. v11c_at50k uses
# v11c's _50k periodic checkpoint to compare apples-to-apples at the OTHER
# end (everyone at ~50k).

set -uo pipefail
ARM="${1:?usage: bash post_qhead_chain_at40k.sh ARM (v13|v14|v15|v16|v17)}"

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

case "$ARM" in
    v13)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v13_jepa_fcstbottleneck128_newconv_fp32_50k_40k.pth"
        FCST_NL=1
        EXTRA_FLAGS="--forecaster-d-model 128 --forecaster-n-heads 4" ;;
    v14)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v14_jepa_enc6_fcst6_dk09_newconv_fp32_50k_40k.pth"
        FCST_NL=6
        EXTRA_FLAGS="" ;;
    v15)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v15_jepa_enc6_fcst4_dk09_newconv_fp32_50k_40k.pth"
        FCST_NL=4
        EXTRA_FLAGS="" ;;
    v16)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v16_jepa_enc6_fcst1_dk07_newconv_fp32_50k_40k.pth"
        FCST_NL=1
        EXTRA_FLAGS="" ;;
    v17)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v17_jepa_enc6_fcst1_dk095_newconv_fp32_50k_40k.pth"
        FCST_NL=1
        EXTRA_FLAGS="" ;;
    v11c_at50k)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v11c_cont_from5k_50k_50k.pth"
        FCST_NL=1
        EXTRA_FLAGS="" ;;
    *)
        echo "ERROR: unknown ARM '$ARM'" >&2
        exit 1 ;;
esac

# Snapshot suffix: at40k for arms v13-v17 (uses _40k.pth), at50k for v11c_at50k
case "$ARM" in
    v11c_at50k) SNAPSHOT_SUFFIX="" ;;  # ARM string already encodes the snapshot
    *)          SNAPSHOT_SUFFIX="_at40k" ;;
esac
RUN_NAME="enc_fcst_${ARM}_qhead_xfmr2L_quant_30k${SNAPSHOT_SUFFIX}"
QHEAD_FINAL="$MAIN/checkpoints/${RUN_NAME}_FINAL.pth"
QHEAD_LOG="$EXP_DIR/results/run_${RUN_NAME}.log"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_${ARM}${SNAPSHOT_SUFFIX}"
TRIAGE_LOG="$EXP_DIR/results/run_triage_${ARM}${SNAPSHOT_SUFFIX}.log"
CHAIN_LOG="$EXP_DIR/results/post_qhead_chain_${ARM}${SNAPSHOT_SUFFIX}.log"

mkdir -p "$EXP_DIR/results" "$TRIAGE_OUT"
echo "=== [${ARM}-at40k chain] START $(date) ===" | tee -a "$CHAIN_LOG"

[ -f "$BB_PATH" ] || { echo "ERROR: backbone _40k missing at $BB_PATH" >&2; exit 1; }

# Phase 1: qhead on the _40k backbone
if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[${ARM}-at40k qhead] launching $RUN_NAME" | tee -a "$CHAIN_LOG"
    PYTHONPATH="$ROOT" \
    HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
    HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
        --backbone-path "$BB_PATH" --forecast-len 16 \
        --quantile-head --head-arch transformer --head-causal true \
        --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
        --head-dropout 0.1 --head-train-input e_then_f \
        --total-steps 30000 --batch-size 256 \
        --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
        --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
        --save-every 5000 --log-every 200 \
        --save-dir "$MAIN/checkpoints" --run-name "$RUN_NAME" \
        --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
        --device cuda \
        --t-raw 4096 --n-channels 1 \
        --d-model 384 --n-heads 6 --num-layers "$FCST_NL" \
        --encoder-type gru \
        $EXTRA_FLAGS \
        --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
        --reconstruction forecaster --amp-dtype bf16 \
        2>&1 | tee -a "$QHEAD_LOG"
    cp -f "$MAIN/checkpoints/${RUN_NAME}_best.pth" "$QHEAD_FINAL"
fi

[ -f "$QHEAD_FINAL" ] || { echo "[${ARM}-at40k qhead] FINAL.pth missing — abort" | tee -a "$CHAIN_LOG"; exit 2; }
echo "[${ARM}-at40k qhead] DONE — proceeding to triage" | tee -a "$CHAIN_LOG"

# Phase 2: triage (same 11 short configs)
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
PYTHONPATH="$ROOT" \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$BB_PATH" \
    --head-path "$QHEAD_FINAL" \
    --output-dir "$TRIAGE_OUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers "$FCST_NL" \
    --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal true \
    $EXTRA_FLAGS \
    --config-filter "$TRIAGE_FILTER" \
    >>"$TRIAGE_LOG" 2>&1

if [ ! -f "$TRIAGE_OUT/summary.txt" ]; then
    echo "[${ARM}-at40k triage] FAILED — no summary" | tee -a "$CHAIN_LOG"
    tail -20 "$TRIAGE_LOG" | tee -a "$CHAIN_LOG"
    exit 3
fi
GM=$(grep -E 'Aggregate GM-Relative MASE' "$TRIAGE_OUT/summary.txt" | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "=== [${ARM}-at40k chain] DONE GM-MASE=${GM:-?} $(date) ===" | tee -a "$CHAIN_LOG"
