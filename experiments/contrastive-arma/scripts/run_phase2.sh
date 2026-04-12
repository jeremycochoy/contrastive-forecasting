#!/bin/bash
# Phase 2: Transformer configuration comparison
# Uses the best encoder from Phase 1 (to be determined — placeholder: BEST_ENCODER)
set -e
cd ~/workspaces/contrastive-forecasting
export CUDA_VISIBLE_DEVICES=1

# Read best encoder from Phase 1 results
BEST_ENCODER=${BEST_ENCODER:-mlp}
echo "Using encoder: $BEST_ENCODER"

COMMON="--H 512 --num-layers 6 --encoder-type $BEST_ENCODER --total-steps 50000 --batch-size 16 --lr 1e-4 --val-every 500 --save-every 25000 --device cuda"

echo "============================================="
echo "Phase 2: Transformer Config Comparison"
echo "============================================="
date

echo ""
echo "=== T1: baseline (nhead=8, ffn=2x, gelu, conv=3) ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 8 --ffn-mult 2 --activation gelu --depthwise-conv 3 \
    --experiment-id T1_baseline --save-path arch_search_T1_baseline.pth

echo ""
echo "=== T2: more heads (nhead=16, head_dim=32) ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 16 --ffn-mult 2 --activation gelu --depthwise-conv 3 \
    --experiment-id T2_heads16 --save-path arch_search_T2_heads16.pth

echo ""
echo "=== T3: FFN 4x ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --experiment-id T3_ffn4x --save-path arch_search_T3_ffn4x.pth

echo ""
echo "=== T4: FFN 1x (TimesFM-like) ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 8 --ffn-mult 1 --activation gelu --depthwise-conv 3 \
    --experiment-id T4_ffn1x --save-path arch_search_T4_ffn1x.pth

echo ""
echo "=== T5: SiLU activation ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 8 --ffn-mult 2 --activation silu --depthwise-conv 3 \
    --experiment-id T5_silu --save-path arch_search_T5_silu.pth

echo ""
echo "=== T6: No depthwise conv ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 8 --ffn-mult 2 --activation gelu --depthwise-conv 0 \
    --experiment-id T6_no_conv --save-path arch_search_T6_no_conv.pth

echo ""
echo "=== T7: Combined best guesses (nhead=16, ffn=2, silu, conv=3) ==="
date
python3 -u train_contrastive_v2.py $COMMON \
    --nhead 16 --ffn-mult 2 --activation silu --depthwise-conv 3 \
    --experiment-id T7_combined --save-path arch_search_T7_combined.pth

echo ""
echo "============================================="
echo "Phase 2 Complete!"
echo "============================================="
date

# Summary
echo ""
echo "=== Phase 2 Results Summary ==="
for id in T1_baseline T2_heads16 T3_ffn4x T4_ffn1x T5_silu T6_no_conv T7_combined; do
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
