#!/bin/bash
# v27c post-q-head chain: promote v27c backbone best_loss→FINAL, train a
# fresh 2L q-head on it → triage. dk0.8 + FFN-only-fp16 backbone.
# Same recipe as post_qhead_chain_v20.sh; GPU pinned to 1 (v20R full-eval
# occupies GPU 0 — do not touch).
set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

BB_BASE="$MAIN/checkpoints/enc_fcst_v27c_dk08_ffnfp16_resume25k_50k"
BB_PATH="${BB_BASE}_FINAL.pth"
RUN_NAME="enc_fcst_v27c_qhead_xfmr2L_quant_30k"
QHEAD_FINAL="$MAIN/checkpoints/${RUN_NAME}_FINAL.pth"
TRIAGE_OUT="$EXP_DIR/results/gift_eval_triage_v27c"
TRIAGE_LOG="$EXP_DIR/results/run_triage_v27c.log"
CHAIN_LOG="$EXP_DIR/results/post_qhead_chain_v27c.log"

mkdir -p "$EXP_DIR/results"
echo "=== [v27c chain] START $(date) ===" | tee -a "$CHAIN_LOG"

# Step 0: promote the v27c backbone's best_loss checkpoint to _FINAL.pth
# (train.py writes _best_loss/_final/_50k — never uppercase _FINAL; the
# q-head is trained on the lowest-contrastive-loss backbone, per project
# convention). Fallback: _final.pth, then _50k.pth.
if [ ! -f "$BB_PATH" ]; then
    if [ -f "${BB_BASE}_best_loss.pth" ]; then BB_SRC="${BB_BASE}_best_loss.pth"
    elif [ -f "${BB_BASE}_final.pth" ]; then BB_SRC="${BB_BASE}_final.pth"
    elif [ -f "${BB_BASE}_50k.pth" ]; then BB_SRC="${BB_BASE}_50k.pth"
    else echo "[v27c chain] no backbone checkpoint to promote — abort" | tee -a "$CHAIN_LOG"; exit 4; fi
    echo "[v27c chain] promote $(basename "$BB_SRC") -> $(basename "$BB_PATH")" | tee -a "$CHAIN_LOG"
    cp -f "$BB_SRC" "$BB_PATH"
fi
[ -f "$BB_PATH" ] || { echo "[v27c chain] backbone FINAL still missing — abort" | tee -a "$CHAIN_LOG"; exit 4; }

# Step 1: train the q-head (foreground). GPU 1 (GPU 0 = v20R full-eval).
if [ ! -f "$QHEAD_FINAL" ]; then
    echo "[v27c qhead] launching $RUN_NAME on GPU 1" | tee -a "$CHAIN_LOG"
    CUDA_VISIBLE_DEVICES=1 bash "$EXP_DIR/scripts/run_qhead_v27c.sh" >>"$CHAIN_LOG" 2>&1
    QHEAD_RC=$?
    if [ "$QHEAD_RC" != 0 ]; then
        echo "[v27c qhead] FAILED rc=$QHEAD_RC — chain aborting" | tee -a "$CHAIN_LOG"
        exit "$QHEAD_RC"
    fi
fi
[ -f "$QHEAD_FINAL" ] || { echo "[v27c qhead] FINAL.pth missing after training — abort" | tee -a "$CHAIN_LOG"; exit 2; }
echo "[v27c qhead] DONE — proceeding to triage" | tee -a "$CHAIN_LOG"

# Step 2: triage on the 11 GIFT-Eval short configs (same filter as v20).
mkdir -p "$TRIAGE_OUT"
TRIAGE_FILTER='bizitobs_application/short|bizitobs_service/short|bizitobs_l2c/(5T|H)/short|ett1/(15T|H)/short|ett2/(15T|H)/short|electricity/H/short|covid_deaths/short|us_births/D/short'

echo "[v27c triage] starting on GPU 1" | tee -a "$CHAIN_LOG"
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
    echo "[ERROR] v27c triage produced no summary.txt — chain aborting" | tee -a "$CHAIN_LOG"
    tail -20 "$TRIAGE_LOG" | tee -a "$CHAIN_LOG"
    exit 3
fi
GM=$(grep -E 'Aggregate GM-Relative MASE' "$SUMMARY" 2>/dev/null \
        | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
PASS=$(awk -v m="${GM:-99}" 'BEGIN{print (m+0 < 1.4369) ? "BEAT_V10" : "WORSE_THAN_V10"}')
echo "[v27c triage] DONE — GM-MASE = ${GM:-?}  ($PASS vs v11c triage 1.388 / v10 1.4369)" | tee -a "$CHAIN_LOG"
echo "=== [v27c chain] DONE $(date) ===" | tee -a "$CHAIN_LOG"
