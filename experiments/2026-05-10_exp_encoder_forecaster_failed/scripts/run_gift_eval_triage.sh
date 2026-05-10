#!/bin/bash
# GIFT-Eval triage (~5 min) for the encoder+forecaster q-head.
#
# Mirrors experiments/2026-05-05_exp_qhead_improvements/scripts/run_eval_elisa.sh
# but points at the encoder+forecaster backbone + its companion 30k q-head
# (auto-detected num_encoder_layers handled in eval_gift_eval_official.py).
#
# Triage filter (matches R9_E13 triage): 11 small-test-set configs that
# total ~5 min wall time. Compare the GM-Relative MASE here against R9_E13's
# triage 0.990 (full eval 1.029, the prior winner).
#
# Output: experiments/2026-05-10_exp_encoder_forecaster/results/gift_eval_triage/
#   all_results.csv  per-config metrics
#   summary.txt      GM-Relative MASE vs seasonal naive + leaderboard
#
# Usage (after the q-head is trained):
#   bash experiments/2026-05-10_exp_encoder_forecaster/scripts/run_gift_eval_triage.sh
#
# Pick a different head with HEAD_NAME=<name> as the env var; default is the
# 30k run name.

set -e

ROOT="/home/jupyter/cf-encoder-forecaster"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

HEAD_NAME="${HEAD_NAME:-enc_fcst_qhead_xfmr12L_quant_30k}"

BB_PATH="$MAIN/checkpoints/enc_fcst_tau_0_10_50k_FINAL.pth"
HEAD_PATH="$MAIN/checkpoints/${HEAD_NAME}_FINAL.pth"
OUT_DIR="$ROOT/experiments/2026-05-10_exp_encoder_forecaster/results/gift_eval_triage"

if [ ! -f "$BB_PATH" ]; then
    echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1
fi
if [ ! -f "$HEAD_PATH" ]; then
    echo "ERROR: head missing at $HEAD_PATH" >&2; exit 1
fi
mkdir -p "$OUT_DIR"

# Pick the most-free 4090 at runtime (training on GPU 1 ought to be done
# before the eval is launched, but we let the live nvidia-smi pick anyway).
GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[eval] using GPU $GPU_FREE for $HEAD_NAME (triage)"

# Same triage filter as run_eval_elisa.sh — 11 configs with <1k test
# instances each, total ~5 min wall time.
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

# Default forecast strategy + length used by the legacy GRU + transformer
# heads in qhead-improvements. The encoder+forecaster backbone is unchanged
# at the head-input contract, so the same settings apply.
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
