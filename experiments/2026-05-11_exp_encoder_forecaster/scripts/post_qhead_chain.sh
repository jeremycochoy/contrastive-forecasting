#!/bin/bash
# Post-q-head chain: triage → gate → full GIFT-Eval (if gate passes).
#
# Polls for the q-head FINAL.pth, then runs triage. Parses triage output
# for the GM-Relative MASE; if < 1.0 launches full GIFT-Eval (all 97
# configs); else exits with a "gate failed" message.
#
# Stdout pattern (consumed by Monitor):
#   [waiting] q-head FINAL.pth not yet present
#   [triage] launched
#   [triage] DONE — GM-MASE = <value>
#   [gate] PASS (< 1.0) → launching full eval
#   [gate] FAIL (>= 1.0) → stopping
#   [full] launched
#   [full] DONE
#   [chain] DONE
#
set -uo pipefail

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

QHEAD_FINAL="$MAIN/checkpoints/enc_fcst_v7_qhead_xfmr12L_quant_30k_FINAL.pth"
QHEAD_LOG="$EXP_DIR/results/run_enc_fcst_v7_qhead_xfmr12L_quant_30k.log"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_v7"

# Wait for q-head to complete (FINAL.pth lands at the end of run_qhead.sh).
# Also bail if q-head log shows a fatal error.
echo "[waiting] q-head FINAL.pth not yet present"
while [ ! -f "$QHEAD_FINAL" ]; do
    if [ -f "$QHEAD_LOG" ] && grep -qE 'Traceback|FAILED|OOM|Killed' "$QHEAD_LOG"; then
        echo "[ERROR] q-head log shows fatal error — chain aborting"
        tail -5 "$QHEAD_LOG"
        exit 1
    fi
    sleep 60
done
echo "[qhead] FINAL.pth landed — proceeding to triage"

# --- Triage ---
echo "[triage] launched"
bash "$EXP_DIR/scripts/run_gift_eval_triage.sh" >>"$EXP_DIR/results/run_triage.log" 2>&1
SUMMARY="$TRIAGE_OUT/summary.txt"
if [ ! -f "$SUMMARY" ]; then
    echo "[ERROR] triage produced no summary.txt — chain aborting"
    exit 2
fi
GM_MASE=$(grep -E 'Aggregate GM-Relative MASE' "$SUMMARY" | head -1 \
              | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [ -z "$GM_MASE" ]; then
    echo "[ERROR] could not parse GM-MASE from $SUMMARY"
    tail -10 "$SUMMARY"
    exit 3
fi
echo "[triage] DONE — GM-MASE = $GM_MASE"

# --- Gate ---
PASS=$(awk -v m="$GM_MASE" 'BEGIN{print (m+0 < 1.0) ? "1" : "0"}')
if [ "$PASS" = "1" ]; then
    echo "[gate] PASS (< 1.0) → launching full eval"
    # Full eval: drop the --config-filter so all 97 configs run.
    HEAD_NAME="enc_fcst_v7_qhead_xfmr12L_quant_30k"
    BB_PATH="$MAIN/checkpoints/enc_fcst_dk09_hsl_b256_fp32_50k_FINAL.pth"
    HEAD_PATH="$MAIN/checkpoints/${HEAD_NAME}_FINAL.pth"
    OUT_DIR="$EXP_DIR/results/gift_eval_full_v7"
    mkdir -p "$OUT_DIR"
    GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free \
                --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
    PYTHONPATH=$ROOT \
    HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
    HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
    GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
    CUDA_VISIBLE_DEVICES="$GPU_FREE" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
        --backbone-path "$BB_PATH" --head-path "$HEAD_PATH" \
        --output-dir "$OUT_DIR" --strategy B4 --forecast-len 16 \
        --t-raw 4096 --backbone-c 1 \
        --d-model 384 --n-heads 6 --num-layers 6 \
        --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
        --head-causal true \
        >>"$EXP_DIR/results/run_full_eval.log" 2>&1
    echo "[full] launched"
    if [ -f "$OUT_DIR/summary.txt" ]; then
        FULL_GM=$(grep -E 'Aggregate GM-Relative MASE' "$OUT_DIR/summary.txt" \
                  | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "[full] DONE — GM-MASE = $FULL_GM"
    else
        echo "[full] DONE — but summary.txt not produced"
    fi
else
    echo "[gate] FAIL (>= 1.0) → stopping; q-head GM-MASE = $GM_MASE"
fi
echo "[chain] DONE"
