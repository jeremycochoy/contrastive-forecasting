#!/bin/bash
# Q-head training on the encoder+forecaster backbone (τ=0.10, 6L+6L).
#
# Mirrors R9_E13's recipe (xfmr 12L causal quantile head, Moirai HP, cosine
# schedule + 2k warmup, e_then_f train input, --reconstruction forecaster)
# but at 30k steps instead of 60k. Rationale (R9_E13 30k vs 60k loss):
#   step 30k mean_loss=0.19310
#   step 45k mean_loss=0.19270
#   step 60k mean_loss=0.19220
# The training loss is essentially flat past 30k (Δ=0.0009 across the second
# half = 0.5% reduction). At RTX 4090 throughput the 30k cut saves ~half the
# wall-clock budget without giving up real signal.
#
# Backbone: encoder+forecaster (6 causal encoder layers + 6 forecaster
# layers, H=384, n_heads=6, GRU patch embed, RevEWMNorm span=128,
# freq+seasonality embeddings dim=3 each, patch_stats=none, τ=0.10).
# Auto-detected by train_forecasting_head.py (num_encoder_layers,
# freq_emb_dim, seasonality_emb_dim, patch_stats).
#
# Output checkpoints land in the MAIN checkout under
# /home/jupyter/contrastive-forecasting/checkpoints/ so the existing sync
# rule (CLAUDE.md) keeps them safe.

set -e

ROOT="/home/jupyter/cf-encoder-forecaster"
MAIN="/home/jupyter/contrastive-forecasting"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=1
# Encoder+forecaster backbone (~22M params) is big; head is 12L too. At
# B=256 the head training forwards through the frozen backbone every step,
# so the activation memory footprint is comparable to the original
# pretraining. expandable_segments avoids fragmentation OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat "$MAIN/experiments/hf_token.txt")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

BB_PATH="$MAIN/checkpoints/enc_fcst_tau_0_10_50k_FINAL.pth"
SAVE_DIR="$MAIN/checkpoints"
RUN_NAME="enc_fcst_qhead_xfmr12L_quant_30k"
LOG_DIR="$ROOT/experiments/2026-05-10_exp_encoder_forecaster/results"
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
    2>&1 | tee -a "$LOG_DIR/run_${RUN_NAME}.log"
cp -f "$SAVE_DIR/${RUN_NAME}_best.pth" "$SAVE_DIR/${RUN_NAME}_FINAL.pth"
echo "=== DONE $RUN_NAME — saved ${RUN_NAME}_FINAL.pth ===" && date
