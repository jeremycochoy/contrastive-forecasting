#!/bin/bash
# v16 JEPA q-head — 2L causal transformer q-head on v16's JEPA backbone.
# v16 differs from v11c only by --encoder-dropkey 0.7 (vs 0.9 in v11c) at
# backbone-pretraining time. Forecaster still 1L, encoder still 6L.
# Q-head architecture + training are IDENTICAL to v11c chain (2L transformer,
# 6 heads, ffn-mult 4, e_then_f, cosine schedule, 2k warmup, bf16, 30k steps).
set -e
ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_v16_jepa_enc6_fcst1_dk07_newconv_fp32_50k_FINAL.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_v16_qhead_xfmr2L_quant_30k"
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

echo "=== START $RUN_NAME (2L qhead on v16 JEPA backbone, fcst=1L, dk=0.7) ===" && date
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
