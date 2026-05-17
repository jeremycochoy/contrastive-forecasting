#!/bin/bash
# v16 orchestrator: waits for backbone Phase 1 to finish (FINAL.pth or
# divergence), then runs Phase 2 (qhead) + Phase 3 (triage) via the chain.
#
# Phase 1 launched separately with nohup. This script polls every 10 min:
#   - If FINAL.pth appears → run post_qhead_chain_v16.sh (which does qhead
#     then triage in foreground).
#   - If process dies without FINAL.pth, or loss > 5 after step 2000, or NaN
#     appears → mark CSV with FAILED_<step> suffix and exit.

set -uo pipefail
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

BB_NAME="enc_fcst_v16_jepa_enc6_fcst1_dk07_newconv_fp32_50k"
BB_FINAL="$MAIN/checkpoints/${BB_NAME}_FINAL.pth"
BB_LOSSES="$MAIN/checkpoints/${BB_NAME}_losses.csv"
BB_PID_FILE="/tmp/v16_backbone.pid"
BB_LOG="$EXP_DIR/results/nohup_v16_backbone.log"
ORCH_LOG="$EXP_DIR/results/orchestrate_v16.log"

mkdir -p "$EXP_DIR/results"
echo "=== [v16 orch] START $(date) ===" | tee -a "$ORCH_LOG"

# Pull PID
BB_PID="$(grep -oE '[0-9]+' "$BB_PID_FILE" | head -1 || true)"
echo "[v16 orch] backbone PID=${BB_PID:-?}" | tee -a "$ORCH_LOG"

# Phase 1: poll until FINAL.pth or failure
while true; do
    if [ -f "$BB_FINAL" ]; then
        echo "[v16 orch] FINAL.pth detected at $(date) — proceeding to Phase 2/3" | tee -a "$ORCH_LOG"
        break
    fi
    # Process death without FINAL → failure
    if [ -n "$BB_PID" ] && ! kill -0 "$BB_PID" 2>/dev/null; then
        echo "[v16 orch] backbone PID $BB_PID died without FINAL.pth at $(date) — failure" | tee -a "$ORCH_LOG"
        LAST_STEP=$(tail -1 "$BB_LOSSES" 2>/dev/null | cut -d, -f1)
        if [ -f "$BB_LOSSES" ]; then
            mv "$BB_LOSSES" "${BB_LOSSES%.csv}_FAILED_${LAST_STEP:-unknown}.csv"
            echo "[v16 orch] renamed CSV → ${BB_LOSSES%.csv}_FAILED_${LAST_STEP:-unknown}.csv" | tee -a "$ORCH_LOG"
        fi
        echo "[v16 orch] === ABORT — Phase 1 failed ===" | tee -a "$ORCH_LOG"
        exit 1
    fi
    # Divergence: loss > 5 after step 2000, or NaN
    if [ -f "$BB_LOSSES" ]; then
        LAST=$(tail -1 "$BB_LOSSES" 2>/dev/null)
        STEP=$(echo "$LAST" | cut -d, -f1)
        LOSS=$(echo "$LAST" | cut -d, -f2)
        if [ -n "$STEP" ] && [ "$STEP" != "step" ]; then
            # NaN check
            if echo "$LOSS" | grep -qiE 'nan|inf'; then
                echo "[v16 orch] NaN/Inf at step $STEP loss=$LOSS — killing PID $BB_PID" | tee -a "$ORCH_LOG"
                [ -n "$BB_PID" ] && kill "$BB_PID" 2>/dev/null || true
                mv "$BB_LOSSES" "${BB_LOSSES%.csv}_FAILED_${STEP}_NaN.csv"
                echo "[v16 orch] === ABORT — Phase 1 NaN ===" | tee -a "$ORCH_LOG"
                exit 2
            fi
            # Divergence after step 2000
            if [ "$STEP" -gt 2000 ] && awk -v l="$LOSS" 'BEGIN{exit !(l+0 > 5.0)}'; then
                echo "[v16 orch] DIVERGED at step $STEP loss=$LOSS (>5 after 2000) — killing PID $BB_PID" | tee -a "$ORCH_LOG"
                [ -n "$BB_PID" ] && kill "$BB_PID" 2>/dev/null || true
                mv "$BB_LOSSES" "${BB_LOSSES%.csv}_FAILED_${STEP}_diverged.csv"
                echo "[v16 orch] === ABORT — Phase 1 diverged ===" | tee -a "$ORCH_LOG"
                exit 3
            fi
            echo "[v16 orch] heartbeat $(date) step=$STEP loss=$LOSS" >> "$ORCH_LOG"
        fi
    fi
    sleep 600
done

# Phase 2 + 3: chain script does qhead training (foreground) then triage
echo "[v16 orch] kicking off post_qhead_chain_v16.sh at $(date)" | tee -a "$ORCH_LOG"
CUDA_VISIBLE_DEVICES=0 bash "$EXP_DIR/scripts/post_qhead_chain_v16.sh" \
    >>"$ORCH_LOG" 2>&1
CHAIN_RC=$?
echo "[v16 orch] post_qhead_chain_v16.sh exit=$CHAIN_RC at $(date)" | tee -a "$ORCH_LOG"
echo "=== [v16 orch] DONE $(date) ===" | tee -a "$ORCH_LOG"
exit $CHAIN_RC
