#!/bin/bash
# GIFT-Eval triage (~5 min) for the encoder+forecaster v2 q-head.
# Triage gate: GM-Relative MASE < 1.0 → full eval; else stop.

set -e

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

HEAD_NAME="${HEAD_NAME:-enc_fcst_dropkey07_qhead_xfmr12L_quant_30k}"

BB_PATH="$MAIN/checkpoints/enc_fcst_dropkey07_pb_50k_FINAL.pth"
HEAD_PATH="$MAIN/checkpoints/${HEAD_NAME}_FINAL.pth"
OUT_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_triage"

if [ ! -f "$BB_PATH" ]; then
    echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1
fi
if [ ! -f "$HEAD_PATH" ]; then
    echo "ERROR: head missing at $HEAD_PATH" >&2; exit 1
fi
mkdir -p "$OUT_DIR"

GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[eval] using GPU $GPU_FREE for $HEAD_NAME (triage)"

TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

FL="${FL:-16}"
STRATEGY="${STRATEGY:-B4}"
HEAD_CAUSAL="${HEAD_CAUSAL:-true}"
echo "[eval] forecast_len=$FL strategy=$STRATEGY head_causal=$HEAD_CAUSAL"

PYTHONPATH=. \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES="$GPU_FREE" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$BB_PATH" \
    --head-path "$HEAD_PATH" \
    --output-dir "$OUT_DIR" --strategy "$STRATEGY" --forecast-len "$FL" \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal "$HEAD_CAUSAL" \
    --config-filter "$TRIAGE_FILTER"

echo "[eval] DONE  out=$OUT_DIR"
if [ -f "$OUT_DIR/summary.txt" ]; then
    echo "--- summary tail ---"
    tail -20 "$OUT_DIR/summary.txt"
fi
