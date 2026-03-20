#!/bin/bash
# Phase 4: Full training with best architecture
# Best config from Phase 1+2: GRU encoder + nhead=8 + FFN 4x + GELU + depthwise_conv=3
set -e
cd ~/workspaces/contrastive-forecasting
export CUDA_VISIBLE_DEVICES=1

echo "============================================="
echo "Phase 4: Full Training - Best Architecture"
echo "============================================="
date

echo ""
echo "Config: GRU encoder, H=1024, 12 layers, nhead=8, FFN 4x, GELU, conv=3"
echo ""

# Full training: 500k steps
# H=1024, 12 layers, FFN 4x => ~80M params
# bs=8 to fit in 24GB with FFN 4x at H=1024
# lr=7e-5 (slightly lower for larger model)
python3 -u train_contrastive_v2.py --device cuda \
    --encoder-type gru --H 1024 --num-layers 12 \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --total-steps 500000 --batch-size 8 --lr 7e-5 \
    --val-every 1000 --save-every 50000 \
    --experiment-id phase4_gru_ffn4x_H1024 \
    --save-path arch_search_phase4_best.pth

echo ""
echo "============================================="
echo "Phase 4 Complete!"
echo "============================================="
date
