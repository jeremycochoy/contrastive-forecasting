#!/bin/bash
# v12 post-q-head chain: train 2L q-head on v12 JEPA backbone → triage.
# Adapted from post_qhead_chain_v10.sh. Encoder is residual_silu instead of gru.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

BB_PATH="$MAIN/checkpoints/enc_fcst_v12_jepa_newconv_residualsilu_fp32_50k_FINAL.pth"
RUN_NAME="enc_fcst_v12_qhead_xfmr2L_quant_30k"
QHEAD_FINAL="$MAIN/checkpoints/${RUN_NAME}_FINAL.pth"
QHEAD_LOG="$EXP_DIR/results/run_${RUN_NAME}.log"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_v12_jepa"
TRIAGE_LOG="$EXP_DIR/results/run_triage_v12_jepa.log"
CHAIN_LOG="$EXP_DIR/results/post_qhead_chain_v12.log"

mkdir -p "$EXP_DIR/results"
echo "=== [v12 chain] START $(date) ===" | tee -a "$CHAIN_LOG"

# Step 1: kick off the q-head training (foreground inside this script).
if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v12 qhead] launching $RUN_NAME on GPU 1" | tee -a "$CHAIN_LOG"
    CUDA_VISIBLE_DEVICES=1 bash "$EXP_DIR/scripts/run_qhead_v12_jepa.sh" \
        >>"$CHAIN_LOG" 2>&1
    QHEAD_RC=$?
    if [ "$QHEAD_RC" != 0 ]; then
        echo "[v12 qhead] FAILED rc=$QHEAD_RC — chain aborting" | tee -a "$CHAIN_LOG"
        exit "$QHEAD_RC"
    fi
fi

if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v12 qhead] FINAL.pth still missing after training — abort" | tee -a "$CHAIN_LOG"
    exit 2
fi
echo "[v12 qhead] DONE — proceeding to triage" | tee -a "$CHAIN_LOG"

# Step 2: triage on 11 GIFT-Eval short configs. Same recipe as v10 chain.
mkdir -p "$TRIAGE_OUT"
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

echo "[v12 triage] starting on GPU 1" | tee -a "$CHAIN_LOG"
PYTHONPATH="$ROOT" \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$BB_PATH" \
    --head-path "$QHEAD_FINAL" \
    --output-dir "$TRIAGE_OUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --encoder-type residual_silu \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal true \
    --config-filter "$TRIAGE_FILTER" \
    >>"$TRIAGE_LOG" 2>&1

SUMMARY="$TRIAGE_OUT/summary.txt"
if [ ! -f "$SUMMARY" ]; then
    echo "[ERROR] triage produced no summary.txt — chain aborting" | tee -a "$CHAIN_LOG"
    tail -20 "$TRIAGE_LOG" | tee -a "$CHAIN_LOG"
    exit 3
fi
GM=$(grep -E 'Aggregate GM-Relative MASE' "$SUMMARY" 2>/dev/null \
        | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PASS=$(awk -v m="${GM:-99}" 'BEGIN{print (m+0 < 1.4369) ? "BEAT_V10" : "WORSE_THAN_V10"}')
echo "[v12 triage] DONE — GM-MASE = ${GM:-?}  ($PASS vs v10=1.4369, v7=1.512)" | tee -a "$CHAIN_LOG"
echo "=== [v12 chain] DONE $(date) ===" | tee -a "$CHAIN_LOG"
