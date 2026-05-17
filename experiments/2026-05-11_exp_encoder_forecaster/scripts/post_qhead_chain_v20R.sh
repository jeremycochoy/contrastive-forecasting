#!/bin/bash
# v20R post-q-head chain: RESUME the v20 2L q-head (from its crashed-at-17k
# _best.pth) → 30k → triage. Same backbone + recipe as post_qhead_chain_v20.sh;
# only run-name (qheadR), launcher (run_qhead_v20R.sh) and output dirs differ.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

BB_PATH="$MAIN/checkpoints/enc_fcst_v20_v11c_freshwarmup_fp16_50k_FINAL.pth"
RUN_NAME="enc_fcst_v20_qheadR_xfmr2L_quant_30k"
QHEAD_FINAL="$MAIN/checkpoints/${RUN_NAME}_FINAL.pth"
QHEAD_LOG="$EXP_DIR/results/run_${RUN_NAME}.log"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_v20R"
TRIAGE_LOG="$EXP_DIR/results/run_triage_v20R.log"
CHAIN_LOG="$EXP_DIR/results/post_qhead_chain_v20R.log"

mkdir -p "$EXP_DIR/results"
echo "=== [v20R chain] START $(date) ===" | tee -a "$CHAIN_LOG"

# Step 1: resume the q-head training (foreground inside this script).
# CUDA_VISIBLE_DEVICES=0 — GPU 1 is occupied by v27c, do not touch.
if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v20R qhead] launching $RUN_NAME on GPU 0" | tee -a "$CHAIN_LOG"
    CUDA_VISIBLE_DEVICES=0 bash "$EXP_DIR/scripts/run_qhead_v20R.sh" \
        >>"$CHAIN_LOG" 2>&1
    QHEAD_RC=$?
    if [ "$QHEAD_RC" != 0 ]; then
        echo "[v20R qhead] FAILED rc=$QHEAD_RC — chain aborting" | tee -a "$CHAIN_LOG"
        exit "$QHEAD_RC"
    fi
fi

if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v20R qhead] FINAL.pth still missing after training — abort" | tee -a "$CHAIN_LOG"
    exit 2
fi
echo "[v20R qhead] DONE — proceeding to triage" | tee -a "$CHAIN_LOG"

# Step 2: triage on 11 GIFT-Eval short configs (same filter as v20 chain).
mkdir -p "$TRIAGE_OUT"
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

echo "[v20R triage] starting on GPU 0" | tee -a "$CHAIN_LOG"
PYTHONPATH="$ROOT" \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$BB_PATH" \
    --head-path "$QHEAD_FINAL" \
    --output-dir "$TRIAGE_OUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --encoder-type gru \
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
echo "[v20R triage] DONE — GM-MASE = ${GM:-?}  ($PASS vs v10=1.4369, v7=1.512)" | tee -a "$CHAIN_LOG"
echo "=== [v20R chain] DONE $(date) ===" | tee -a "$CHAIN_LOG"
