#!/bin/bash
# v17 post-q-head chain: train 2L q-head on v17 JEPA backbone (6L enc + 1L fcst,
# dk=0.95) → triage on 11 GIFT-Eval short configs.
#
# Adapted from post_qhead_chain_v15.sh — only paths/names updated and
# GPU pinned to 1 (v16 backbone is on GPU 0, do not touch).
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

BB_PATH="$MAIN/checkpoints/enc_fcst_v17_jepa_enc6_fcst1_dk095_newconv_fp32_50k_FINAL.pth"
RUN_NAME="enc_fcst_v17_qhead_xfmr2L_quant_30k"
QHEAD_FINAL="$MAIN/checkpoints/${RUN_NAME}_FINAL.pth"
QHEAD_LOG="$EXP_DIR/results/run_${RUN_NAME}.log"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_v17_jepa"
TRIAGE_LOG="$EXP_DIR/results/run_triage_v17.log"
CHAIN_LOG="$EXP_DIR/results/post_qhead_chain_v17.log"

mkdir -p "$EXP_DIR/results"
echo "=== [v17 chain] START $(date) ===" | tee -a "$CHAIN_LOG"

# Step 1: q-head training. Pin GPU 1 (v16 lives on GPU 0).
if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v17 qhead] launching $RUN_NAME on GPU 1" | tee -a "$CHAIN_LOG"
    CUDA_VISIBLE_DEVICES=1 bash "$EXP_DIR/scripts/run_qhead_v17.sh" \
        >>"$CHAIN_LOG" 2>&1
    QHEAD_RC=$?
    if [ "$QHEAD_RC" != 0 ]; then
        echo "[v17 qhead] FAILED rc=$QHEAD_RC — chain aborting" | tee -a "$CHAIN_LOG"
        exit "$QHEAD_RC"
    fi
fi

if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v17 qhead] FINAL.pth still missing after training — abort" | tee -a "$CHAIN_LOG"
    exit 2
fi
echo "[v17 qhead] DONE — proceeding to triage" | tee -a "$CHAIN_LOG"

# Step 2: triage on 11 GIFT-Eval short configs. Same recipe as v11c/v15 chain.
mkdir -p "$TRIAGE_OUT"
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

echo "[v17 triage] starting on GPU 1" | tee -a "$CHAIN_LOG"
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
PASS=$(awk -v m="${GM:-99}" 'BEGIN{print (m+0 < 1.388) ? "BEAT_V11C" : "WORSE_THAN_V11C"}')
echo "[v17 triage] DONE — GM-MASE = ${GM:-?}  ($PASS vs v11c=1.388, v14=1.650, v15=1.671)" | tee -a "$CHAIN_LOG"
echo "=== [v17 chain] DONE $(date) ===" | tee -a "$CHAIN_LOG"
