#!/bin/bash
# Phase 5: Parameter recovery validation on best Phase 4 model
# Uses DeepGRU recovery head (best from previous experiments)
set -e
cd ~/workspaces/contrastive-forecasting
export CUDA_VISIBLE_DEVICES=1

echo "============================================="
echo "Phase 5: Parameter Recovery Validation"
echo "============================================="
date

# Wait for Phase 4 if still running
while pgrep -f "train_contrastive_v2.*phase4" > /dev/null 2>&1; do
    echo "Waiting for Phase 4 to finish..."
    sleep 300
done

echo "Phase 4 done. Starting parameter recovery."

# Use the best checkpoint from Phase 4
MODEL_PATH="arch_search_phase4_best_best.pth"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="arch_search_phase4_best.pth"
fi
echo "Using backbone: $MODEL_PATH"

# Train DeepGRU recovery head using the new configurable model
python3 -u train_parameter_recovery_v2.py \
    --device cuda \
    --model-path "$MODEL_PATH" \
    --encoder-type gru --H 1024 --num-layers 12 \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --model-type deepgru \
    --hidden-dim 256 \
    --num-arma-params 4 \
    --dimension 4 \
    --epochs 20000 \
    --batch-size 32 \
    --lr 1e-3 \
    --log-every 100 \
    --save-every 5000 \
    --head-path phase5_recovery_deepgru.pth

echo ""
echo "=== Evaluating recovery ==="
python3 -u train_parameter_recovery_v2.py \
    --device cuda \
    --model-path "$MODEL_PATH" \
    --encoder-type gru --H 1024 --num-layers 12 \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --model-type deepgru \
    --hidden-dim 256 \
    --num-arma-params 4 \
    --dimension 4 \
    --evaluate \
    --head-path phase5_recovery_deepgru.pth \
    --eval-samples 200

echo ""
echo "============================================="
echo "Phase 5 Complete!"
echo "============================================="
date
