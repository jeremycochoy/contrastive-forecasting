#!/bin/bash
# v11c JEPA q-head — 2L causal transformer q-head on v11c's JEPA backbone
# (encoder=6L, forecaster=1L; NEW conv placement: --depthwise-conv 3,
# --deprecated-depthwise-conv 0; PURE fp32 body).
#
# Same recipe as the v10 chain (R9_E13 + e_then_f + Moirai HP + cosine
# + 2k warmup, bf16, 30k steps). Only differences from run_qhead_v10_jepa.sh:
#   • backbone path = v11c FINAL (50k cont-from-5k)
#   • run name suffix = v11c (vs v10jepa)
#   • GPU 0 default (vs 1) — v12 occupies GPU 1
#
# BACKBONE_CONFIG default depthwise_conv=3 → NEW placement at inference,
# matching v11c's training-time graph. No extra placement flags needed.
#
# Compare triage GM-MASE to v10 (1.4369) and v7 (1.512). User hypothesis:
# v11c's clean-residual encoder produces genuinely semantic features → MASE
# should IMPROVE despite v11c's higher contrastive loss (~2.10 vs v10's 1.45).
#
# Run name: enc_fcst_v11c_qhead_xfmr2L_quant_30k.
set -e
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_v11c_cont_from5k_50k_FINAL.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_v11c_qhead_xfmr2L_quant_30k"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"

[ -f "$BB_PATH" ] || { echo "ERROR: backbone missing at $BB_PATH" >&2; exit 1; }
[ -f "$SAVE_DIR/${RUN_NAME}_FINAL.pth" ] && { echo "=== SKIP — exists ==="; exit 0; }
# safety: don't collide with any existing checkpoint (project rule)
for suf in _best.pth _final.pth _losses.csv _5k.pth; do
    if [ -e "$SAVE_DIR/${RUN_NAME}${suf}" ]; then
        echo "ERROR: ${RUN_NAME}${suf} already exists — refusing to clobber" >&2
        exit 1
    fi
done

echo "=== START $RUN_NAME (2L qhead on v11c JEPA backbone) ===" && date
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 2 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$RUN_NAME" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 1 \
    --encoder-type gru \
    --mix-ratio 0.0 --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster --amp-dtype bf16 \
    2>&1 | tee -a "$LOG_DIR/run_${RUN_NAME}.log"
cp -f "$SAVE_DIR/${RUN_NAME}_best.pth" "$SAVE_DIR/${RUN_NAME}_FINAL.pth"
echo "=== DONE $RUN_NAME ===" && date
