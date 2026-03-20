#!/bin/bash
# Phase 3: Scaling experiments
# Uses best encoder (GRU) and best transformer config from Phase 2
set -e
cd ~/workspaces/contrastive-forecasting
export CUDA_VISIBLE_DEVICES=1

ENCODER=${ENCODER:-gru}
NHEAD=${NHEAD:-8}
FFN_MULT=${FFN_MULT:-2}
ACTIVATION=${ACTIVATION:-gelu}
CONV_K=${CONV_K:-3}

echo "Using: encoder=$ENCODER nhead=$NHEAD ffn_mult=$FFN_MULT act=$ACTIVATION conv=$CONV_K"

echo "============================================="
echo "Phase 3: Scaling Experiments"
echo "============================================="
date

echo ""
echo "=== S1: 12L H=1024, nhead=$NHEAD (current scale) ==="
date
python3 -u train_contrastive_v2.py --device cuda \
    --encoder-type $ENCODER --H 1024 --num-layers 12 \
    --nhead $NHEAD --ffn-mult $FFN_MULT --activation $ACTIVATION --depthwise-conv $CONV_K \
    --total-steps 100000 --batch-size 16 --lr 1e-4 --val-every 500 --save-every 25000 \
    --experiment-id S1_12L_H1024 --save-path arch_search_S1_12L_H1024.pth

echo ""
echo "=== S2: 16L H=1024 ==="
date
python3 -u train_contrastive_v2.py --device cuda \
    --encoder-type $ENCODER --H 1024 --num-layers 16 \
    --nhead $NHEAD --ffn-mult $FFN_MULT --activation $ACTIVATION --depthwise-conv $CONV_K \
    --total-steps 100000 --batch-size 16 --lr 1e-4 --val-every 500 --save-every 25000 \
    --experiment-id S2_16L_H1024 --save-path arch_search_S2_16L_H1024.pth

echo ""
echo "=== S3: 20L H=1280 (TimeFM-like) ==="
date
python3 -u train_contrastive_v2.py --device cuda \
    --encoder-type $ENCODER --H 1280 --num-layers 20 \
    --nhead 16 --ffn-mult $FFN_MULT --activation $ACTIVATION --depthwise-conv $CONV_K \
    --total-steps 100000 --batch-size 8 --lr 7e-5 --val-every 500 --save-every 25000 \
    --experiment-id S3_20L_H1280 --save-path arch_search_S3_20L_H1280.pth

echo ""
echo "=== S4: 12L H=1280 ==="
date
python3 -u train_contrastive_v2.py --device cuda \
    --encoder-type $ENCODER --H 1280 --num-layers 12 \
    --nhead 16 --ffn-mult $FFN_MULT --activation $ACTIVATION --depthwise-conv $CONV_K \
    --total-steps 100000 --batch-size 12 --lr 8e-5 --val-every 500 --save-every 25000 \
    --experiment-id S4_12L_H1280 --save-path arch_search_S4_12L_H1280.pth

echo ""
echo "============================================="
echo "Phase 3 Complete!"
echo "============================================="
date

echo ""
echo "=== Phase 3 Results Summary ==="
for id in S1_12L_H1024 S2_16L_H1024 S3_20L_H1280 S4_12L_H1280; do
    f="arch_search_${id}_results.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
fm = d.get('final_metrics', {})
print(f\"  {d['experiment_id']:20s} | FF={fm.get('val_ff',0):.4f} FP={fm.get('val_fp',0):.4f} gap={fm.get('val_ff_fp_gap',0):.4f} CB={fm.get('val_cb',0):.4f} | best_FF={d['best_val_ff']:.4f}@{d['best_step']} | {d['n_params']:,} params | {d['total_time_sec']/60:.0f}min\")
"
    fi
done
