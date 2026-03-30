#!/bin/bash
# Phase 1: Encoder comparison (H=512, 6 layers, 50k steps each)
set -e
cd ~/workspaces/contrastive-forecasting
export CUDA_VISIBLE_DEVICES=1

COMMON="--H 512 --num-layers 6 --nhead 8 --ffn-mult 2 --total-steps 50000 --batch-size 16 --lr 1e-4 --val-every 500 --save-every 25000 --device cuda"

echo "============================================="
echo "Phase 1: Encoder Comparison"
echo "============================================="
date

echo ""
echo "=== E1: MLP baseline (intermediate=64) ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --encoder-type mlp --intermediate-dim 64 \
    --experiment-id E1_mlp --save-path arch_search_E1_mlp.pth

echo ""
echo "=== E2: MLP wide (intermediate=256) ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --encoder-type mlp_wide --intermediate-dim 256 \
    --experiment-id E2_mlp_wide --save-path arch_search_E2_mlp_wide.pth

echo ""
echo "=== E3: Residual SiLU ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --encoder-type residual_silu \
    --experiment-id E3_residual_silu --save-path arch_search_E3_residual_silu.pth

echo ""
echo "=== E4: GRU encoder ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --encoder-type gru \
    --experiment-id E4_gru --save-path arch_search_E4_gru.pth

echo ""
echo "=== E5: Conv encoder ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --encoder-type conv \
    --experiment-id E5_conv --save-path arch_search_E5_conv.pth

echo ""
echo "============================================="
echo "Phase 1 Complete!"
echo "============================================="
date

# Summary
echo ""
echo "=== Phase 1 Results Summary ==="
for id in E1_mlp E2_mlp_wide E3_residual_silu E4_gru E5_conv; do
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
