#!/bin/bash
# Local GIFT-Eval driver for the qhead-improvements Round 1+ heads.
# Eval runs on elisa (free 4090s, GIFT-Eval data already on disk) — vast just
# does training.
#
# Usage:
#   [FL=N] [STRATEGY=S] [HEAD_CAUSAL={true,false}] \
#       ./run_eval_elisa.sh <head_run_name> [--triage|--full]
#
# Defaults (matching the legacy GRU and linear-probe heads):
#   FL=16, STRATEGY=B4, HEAD_CAUSAL=true.
#
# For a bidir / forecast_len=128 head (e.g. R6_E8) you must pass
# overrides explicitly — there's no name-based magic:
#   FL=128 STRATEGY=B1 HEAD_CAUSAL=false ./run_eval_elisa.sh \
#       R6_E8_xfmr6L_bidir_fl128_quant_moirai_cosine --triage

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
        # Fast triage set — picks small-test-set configs only.
        # Earlier attempt that included m4_yearly took 20+min because
        # m4_yearly has ~23k test instances. Restricted now to configs
        # with <1k instances. Filter regex matches against '<dataset>/<term>'.
        EXTRA+=(--config-filter 'bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short')
        ;;
    --full)
        : # all 97 configs
        ;;
    *)
        echo "ERROR: unknown mode $MODE" >&2
        exit 2
        ;;
esac

# Explicit env-var overrides — no name-based magic. Defaults match the
# legacy GRU + linear-probe heads.
FL="${FL:-16}"
STRATEGY="${STRATEGY:-B4}"
HEAD_CAUSAL="${HEAD_CAUSAL:-true}"
echo "[eval] forecast_len=$FL strategy=$STRATEGY head_causal=$HEAD_CAUSAL"

PYTHONPATH=. \
HF_TOKEN=$(cat experiments/hf_token.txt) \
HUGGING_FACE_HUB_TOKEN=$(cat experiments/hf_token.txt) \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
CUDA_VISIBLE_DEVICES=${GPU_FREE} \
python3 -u experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "$BB_PATH" \
    --head-path "$HEAD_PATH" \
    --output-dir "$OUT_DIR" --strategy "$STRATEGY" --forecast-len "$FL" \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal "$HEAD_CAUSAL" \
    "${EXTRA[@]}"

echo "[eval] DONE  out=$OUT_DIR"
if [ -f "$OUT_DIR/summary.txt" ]; then
    echo "--- summary tail ---"
    tail -10 "$OUT_DIR/summary.txt"
fi
