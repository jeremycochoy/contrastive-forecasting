#!/bin/bash
# Generic full-GIFT-Eval (no --config-filter) launcher for the 2L-head sweep.
# Usage:  CUDA_VISIBLE_DEVICES=N bash run_full_eval_2L.sh ARM
# where ARM ∈ {v11c, v14, v15, v16, v17}. Per-arm config (backbone path,
# qhead path, --num-layers) is set inside this script. Output goes to
# experiments/2026-05-11_exp_encoder_forecaster/results/gift_eval_full_<ARM>/.
#
# Triage was on 11 short configs; full eval runs all configs (med/long too).

set -uo pipefail
ARM="${1:?usage: bash run_full_eval_2L.sh ARM (v11c|v14|v15|v16|v17)}"

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
EXP_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster"

case "$ARM" in
    v11c)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v11c_cont_from5k_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v11c_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    v11c_at50k)
        # v11c's step-50k periodic backbone + the fresh 2L qhead trained on it
        # (the "last v11c re-run"). Fills the v11c full-GM @50k gap (we only
        # had triage 1.365 here; full @40.6k was 1.292).
        BB_PATH="$MAIN/checkpoints/enc_fcst_v11c_cont_from5k_50k_50k.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v11c_at50k_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    v20)
        # fp32-warmup→fp16 recipe validation (held to 50k stable).
        BB_PATH="$MAIN/checkpoints/enc_fcst_v20_v11c_freshwarmup_fp16_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v20_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    v20R)
        # same v20 backbone; q-head RESUMED from the crashed-at-17k _best.pth.
        BB_PATH="$MAIN/checkpoints/enc_fcst_v20_v11c_freshwarmup_fp16_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v20_qheadR_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    v27c)
        # dk0.8 + FFN-only-fp16 no-warmup backbone (resumed-from-clean-25k→50k).
        BB_PATH="$MAIN/checkpoints/enc_fcst_v27c_dk08_ffnfp16_resume25k_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v27c_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    v13)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v13_jepa_fcstbottleneck128_newconv_fp32_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v13_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1
        # v13 has bottlenecked forecaster — must pass these flags through
        EXTRA_FLAGS="--forecaster-d-model 128 --forecaster-n-heads 4" ;;
    v14)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v14_jepa_enc6_fcst6_dk09_newconv_fp32_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v14_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=6 ;;
    v15)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v15_jepa_enc6_fcst4_dk09_newconv_fp32_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v15_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=4 ;;
    v16)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v16_jepa_enc6_fcst1_dk07_newconv_fp32_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v16_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    v17)
        BB_PATH="$MAIN/checkpoints/enc_fcst_v17_jepa_enc6_fcst1_dk095_newconv_fp32_50k_FINAL.pth"
        QHEAD="$MAIN/checkpoints/enc_fcst_v17_qhead_xfmr2L_quant_30k_FINAL.pth"
        FCST_NL=1 ;;
    *)
        echo "ERROR: unknown ARM '$ARM'. choices: v11c, v14, v15, v16, v17" >&2
        exit 1 ;;
esac

OUT="$EXP_DIR/results/gift_eval_full_${ARM}"
LOG="$EXP_DIR/results/run_full_eval_${ARM}.log"
mkdir -p "$OUT"
[ -f "$BB_PATH" ] || { echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1; }
[ -f "$QHEAD" ]   || { echo "ERROR: qhead missing at $QHEAD"   >&2; exit 1; }
[ -f "$OUT/summary.txt" ] && { echo "=== SKIP — $OUT/summary.txt exists ==="; exit 0; }

echo "=== START full GM-MASE eval $ARM ($(date)) ===" | tee -a "$LOG"
PYTHONPATH="$ROOT" \
HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
HUGGING_FACE_HUB_TOKEN=$(cat "$MAIN/experiments/hf_token.txt") \
GIFT_EVAL=/home/jupyter/workspaces/gift-eval-data \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py" \
    --backbone-path "$BB_PATH" \
    --head-path "$QHEAD" \
    --output-dir "$OUT" --strategy B4 --forecast-len 16 \
    --t-raw 4096 --backbone-c 1 \
    --d-model 384 --n-heads 6 --num-layers "$FCST_NL" \
    --encoder-type gru \
    --rev-norm-kind ewma --rev-norm-span 128 --device cuda \
    --head-causal true \
    ${EXTRA_FLAGS:-} \
    >>"$LOG" 2>&1
RC=$?
if [ "$RC" != 0 ] || [ ! -f "$OUT/summary.txt" ]; then
    echo "[ERROR] $ARM full eval failed rc=$RC; tail of log follows" | tee -a "$LOG"
    tail -20 "$LOG"
    exit "$RC"
fi
GM=$(grep -E 'Aggregate GM-Relative MASE' "$OUT/summary.txt" 2>/dev/null \
        | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "=== DONE full GM-MASE eval $ARM = ${GM:-?}  ($(date)) ===" | tee -a "$LOG"
