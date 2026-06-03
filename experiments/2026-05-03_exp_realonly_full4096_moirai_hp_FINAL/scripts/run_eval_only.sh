#!/bin/bash
# STAGE E only — re-run GIFT-eval with GIFT_EVAL env exported.
# The first attempt in run_qhead_eval.sh failed all 97 configs with
# "argument should be a str or PathLike, not NoneType" because the
# GIFT_EVAL env var wasn't exported. Backbone and qhead checkpoints
# are unaffected; this just re-runs the eval.

set -e
cd /workspace/app

export PYTHONPATH=/workspace/app
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export GIFT_EVAL=/workspace/gift-eval-data

BB="tiny_full4096_moirai_hp_FRESH_RESUME50k"
QH="R1q_full4096_moirai_hp_FRESH_RESUME50k"
RES_DIR="experiments/2026-05-03_2026-05-02_exp_realonly_full4096_moirai_hp_FINAL/results/gift_eval_resume50k"
mkdir -p "$RES_DIR"

echo "" && echo "=== STAGE E (retry): gift_eval ===" && date
python3 -u experiments/gift-eval/scripts/eval_gift_eval_official.py \
    --backbone-path "checkpoints/${BB}_FINAL.pth" \
    --head-path "checkpoints/${QH}_FINAL.pth" \
    --output-dir "$RES_DIR" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda
echo "=== STAGE E DONE ===" && date
echo "" && echo "=== run_full4096_moirai_hp_FRESH_RESUME50k: ALL DONE ===" && date

if [ -f "$RES_DIR/summary.txt" ]; then
    cat "$RES_DIR/summary.txt"
fi
