#!/bin/bash
# Local GIFT-Eval driver for the qhead-improvements Round 1 heads.
# Eval runs on elisa (free 4090s, GIFT-Eval data already on disk) — vast just
# does training. Usage:
#   ./run_eval_elisa.sh <head_run_name> [--triage|--full]
# Examples:
#   ./run_eval_elisa.sh R1_E1_linear_quant_lr3e4 --triage
#   ./run_eval_elisa.sh R1_E2_gru_quant_moirai_wsd --full

set -e
ROOT="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

HEAD_NAME="${1:?usage: $0 <head_run_name> [--triage|--full]}"
MODE="${2:---full}"

BB_PATH="${ROOT}/sync_realonly_full4096_moirai_hp_FRESH_RESUME50k/moirai_hp_FRESH_RESUME50k/checkpoints/tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth"
HEAD_PATH="${ROOT}/sync_qhead_beta_rd1/checkpoints/${HEAD_NAME}_FINAL.pth"
OUT_DIR="${ROOT}/experiments/2026-05-05_exp_qhead_improvements/results/${HEAD_NAME}_${MODE#--}"

if [ ! -f "$BB_PATH" ]; then echo "ERROR: backbone-beta missing at $BB_PATH" >&2; exit 1; fi
if [ ! -f "$HEAD_PATH" ]; then echo "ERROR: head missing at $HEAD_PATH" >&2; exit 1; fi
mkdir -p "$OUT_DIR"

# Pick the most-free 4090 at runtime.
GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "[eval] using GPU ${GPU_FREE} for $HEAD_NAME ($MODE)"

EXTRA=()
case "$MODE" in
    --triage)
        # 5-config fast set (~90-120s) — picks 1 small config per major domain.
        # Per the eval-script investigation: filter by '<dataset>/<term>'.
        EXTRA+=(--config-filter 'bizitobs_application/10S/short|ett1/15T/short|m4_yearly/A/short|covid_deaths/D/short|electricity/H/short')
        ;;
    --full)
        : # all 97 configs
        ;;
    *)
        echo "ERROR: unknown mode $MODE" >&2
        exit 2
        ;;
esac

PYTHONPATH=. \
HF_TOKEN=$(cat experiments/hf_token.txt) \
HUGGING_FACE_HUB_TOKEN=$(cat experiments/hf_token.txt) \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES=${GPU_FREE} \
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "$BB_PATH" \
    --head-path "$HEAD_PATH" \
    --output-dir "$OUT_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    "${EXTRA[@]}"

echo "[eval] DONE  out=$OUT_DIR"
if [ -f "$OUT_DIR/summary.txt" ]; then
    echo "--- summary tail ---"
    tail -10 "$OUT_DIR/summary.txt"
fi
