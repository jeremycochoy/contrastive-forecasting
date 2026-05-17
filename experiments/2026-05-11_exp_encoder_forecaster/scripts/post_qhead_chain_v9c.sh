#!/bin/bash
# v9c post-q-head chain: triage → gate.
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

QHEAD_FINAL="$MAIN/checkpoints/enc_fcst_v9c_qhead_xfmr12L_quant_30k_FINAL.pth"
QHEAD_LOG="$EXP_DIR/results/run_enc_fcst_v9c_qhead_xfmr12L_quant_30k.log"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_v9c"

echo "[waiting] v9c q-head FINAL.pth not yet present"
while [ ! -f "$QHEAD_FINAL" ]; do
    if [ -f "$QHEAD_LOG" ] && grep -qE 'Traceback|FAILED|OOM' "$QHEAD_LOG"; then
        echo "[ERROR] v9c q-head log shows fatal error"; tail -5 "$QHEAD_LOG"; exit 1
    fi
    sleep 60
done
echo "[v9c qhead] FINAL.pth landed — proceeding to triage"

mkdir -p "$TRIAGE_OUT"
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'
GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)

PYTHONPATH=$ROOT \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES="$GPU_FREE" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$MAIN/checkpoints/enc_fcst_v7_continue_100k_FINAL.pth" \
    --head-path "$QHEAD_FINAL" \
    --output-dir "$TRIAGE_OUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal true \
    --config-filter "$TRIAGE_FILTER" \
    >>"$EXP_DIR/results/run_triage_v9c.log" 2>&1

SUMMARY="$TRIAGE_OUT/summary.txt"
GM=$(grep -E 'Aggregate GM-Relative MASE' "$SUMMARY" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PASS=$(awk -v m="${GM:-99}" 'BEGIN{print (m+0 < 1.0) ? "PASS" : "FAIL"}')
echo "[v9c triage] DONE — GM-MASE = ${GM:-?} ($PASS gate, vs v7=1.512)"
echo "[v9c chain] DONE"
