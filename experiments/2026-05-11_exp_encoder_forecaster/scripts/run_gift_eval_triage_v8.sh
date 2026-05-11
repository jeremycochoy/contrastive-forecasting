#!/bin/bash
# GIFT-Eval triage on v8's backbone + q-head.
set -e

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

HEAD_NAME="${HEAD_NAME:-enc_fcst_v8_qhead_xfmr12L_quant_30k}"
BB_PATH="$MAIN/checkpoints/enc_fcst_dk07_pb_fp32_resume_50k_FINAL.pth"
HEAD_PATH="$MAIN/checkpoints/${HEAD_NAME}_FINAL.pth"
OUT_DIR="${OUT_DIR_OVERRIDE:-$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_triage_v8}"

if [ ! -f "$BB_PATH" ]; then echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1; fi
if [ ! -f "$HEAD_PATH" ]; then echo "ERROR: head missing at $HEAD_PATH" >&2; exit 1; fi
mkdir -p "$OUT_DIR"

GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[v8 eval] using GPU $GPU_FREE for $HEAD_NAME"

TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

PYTHONPATH=. \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES="$GPU_FREE" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$BB_PATH" \
    --head-path "$HEAD_PATH" \
    --output-dir "$OUT_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal true \
    --config-filter "$TRIAGE_FILTER"

echo "[v8 eval] DONE  out=$OUT_DIR"
[ -f "$OUT_DIR/summary.txt" ] && tail -20 "$OUT_DIR/summary.txt"
