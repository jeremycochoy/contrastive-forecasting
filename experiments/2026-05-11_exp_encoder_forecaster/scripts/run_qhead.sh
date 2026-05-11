#!/bin/bash
# Q-head training on the encoder+forecaster v2 backbone (dropkey=0.7, 6L+6L).
#
# Mirrors R9_E13's recipe (xfmr 12L causal quantile head, Moirai HP, cosine
# schedule + 2k warmup, e_then_f train input, --reconstruction forecaster).
# 30k steps, --amp-dtype bf16 per PLAN.md.
#
# Backbone: encoder+forecaster v2 (6 causal encoder layers + 6 forecaster
# layers, H=384, n_heads=6, GRU patch embed, RevEWMNorm span=128,
# freq+seasonality embeddings dim=3 each, patch_stats=none, τ=0.10,
# encoder-dropkey=0.7).

set -e

ROOT="/home/jupyter/cf-encoder-forecaster-v2"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_dropkey07_BACKBONE_step10200_FINAL.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_dropkey07_qhead_xfmr12L_quant_30k"
LOG_DIR="$ROOT/experiments/2026-05-11_exp_encoder_forecaster/results"
mkdir -p "$LOG_DIR"

if [ ! -f "$BB_PATH" ]; then
    echo "ERROR: backbone missing at $BB_PATH" >&2
    exit 1
fi
if [ -f "$SAVE_DIR/${RUN_NAME}_FINAL.pth" ]; then
    echo "=== SKIP — $SAVE_DIR/${RUN_NAME}_FINAL.pth exists ==="
    exit 0
fi

echo "=== START $RUN_NAME ===" && date
python3 -u "$ROOT/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py" \
    --backbone-path "$BB_PATH" --forecast-len 16 \
    --quantile-head --head-arch transformer --head-causal true \
    --head-num-layers 12 --head-nhead 6 --head-ffn-mult 4.0 \
    --head-dropout 0.1 \
    --head-train-input e_then_f \
    --total-steps 30000 --batch-size 256 \
    --lr 1e-3 --beta1 0.9 --beta2 0.98 --weight-decay 0.1 \
    --schedule cosine --warmup-steps 2000 --final-lr-ratio 0.1 \
    --save-every 5000 --log-every 200 \
    --save-dir "$SAVE_DIR" --run-name "$RUN_NAME" \
    --hf-repo "jeremycochoy/gift-pretrain-full-4096" --hf-path "small_v1" \
    --device cuda \
    --t-raw 4096 --n-channels 1 \
    --d-model 384 --n-heads 6 --num-layers 6 \
    --mix-ratio 0.0 \
    --rev-norm-kind ewma --rev-norm-span 128 \
    --reconstruction forecaster \
    --amp-dtype bf16 \
    2>&1 | tee -a "$LOG_DIR/run_${RUN_NAME}.log"
cp -f "$SAVE_DIR/${RUN_NAME}_best.pth" "$SAVE_DIR/${RUN_NAME}_FINAL.pth"
echo "=== DONE $RUN_NAME — saved ${RUN_NAME}_FINAL.pth ===" && date
