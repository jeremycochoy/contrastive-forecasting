#!/bin/bash
# Run all parameter recovery experiments sequentially on GPU 1
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_experiments.sh

set -e
cd ~/workspaces/contrastive-forecasting

echo "=================================================="
echo "Starting ARMA Parameter Recovery Experiments"
echo "=================================================="
date

# Experiment 1: MLP baseline on H1024 (already running separately, skip if exists)
if [ ! -f "parameter_recovery_mlp_h1024_best.pth" ]; then
    echo ""
    echo "=== Experiment 1: MLP baseline (H1024, 30k epochs) ==="
    date
    python3 train_parameter_recovery.py \
        --model-type mlp --device cuda \
        --epochs 30000 --batch-size 32 --lr 1e-3 \
        --hidden-dim 256 \
        --head-path parameter_recovery_mlp_h1024.pth \
        --log-every 1000 --save-every 5000 \
        2>&1 | tee exp_mlp.log
fi

# Experiment 2: GRU model
echo ""
echo "=== Experiment 2: GRU (H1024, 30k epochs) ==="
date
python3 train_parameter_recovery.py \
    --model-type gru --device cuda \
    --epochs 30000 --batch-size 32 --lr 5e-4 \
    --hidden-dim 256 \
    --head-path parameter_recovery_gru.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_gru.log

# Experiment 3: Residual MLP
echo ""
echo "=== Experiment 3: Residual MLP (H1024, 30k epochs) ==="
date
python3 train_parameter_recovery.py \
    --model-type resmlp --device cuda \
    --epochs 30000 --batch-size 32 --lr 5e-4 \
    --hidden-dim 256 \
    --head-path parameter_recovery_resmlp.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_resmlp.log

# Experiment 4: Attention
echo ""
echo "=== Experiment 4: Attention (H1024, 30k epochs) ==="
date
python3 train_parameter_recovery.py \
    --model-type attention --device cuda \
    --epochs 30000 --batch-size 32 --lr 5e-4 \
    --hidden-dim 256 \
    --head-path parameter_recovery_attention.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_attention.log

# Experiment 5: GRU with pooling
echo ""
echo "=== Experiment 5: GRU Pool (H1024, 30k epochs) ==="
date
python3 train_parameter_recovery.py \
    --model-type grupool --device cuda \
    --epochs 30000 --batch-size 32 --lr 5e-4 \
    --hidden-dim 256 \
    --head-path parameter_recovery_grupool.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_grupool.log

# Experiment 6: GRU with larger hidden dim
echo ""
echo "=== Experiment 6: GRU large hidden (H1024, 30k epochs, hidden=512) ==="
date
python3 train_parameter_recovery.py \
    --model-type gru --device cuda \
    --epochs 30000 --batch-size 32 --lr 3e-4 \
    --hidden-dim 512 \
    --head-path parameter_recovery_gru_large.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_gru_large.log

# Experiment 7: MLP with AdamW and cosine schedule
echo ""
echo "=== Experiment 7: MLP + AdamW + cosine schedule ==="
date
python3 train_parameter_recovery.py \
    --model-type mlp --device cuda \
    --epochs 30000 --batch-size 32 --lr 1e-3 \
    --hidden-dim 256 \
    --optimizer adamw --weight-decay 0.01 --scheduler cosine \
    --head-path parameter_recovery_mlp_cosine.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_mlp_cosine.log

# Experiment 8: GRU with cosine schedule
echo ""
echo "=== Experiment 8: GRU + cosine schedule ==="
date
python3 train_parameter_recovery.py \
    --model-type gru --device cuda \
    --epochs 30000 --batch-size 32 --lr 1e-3 \
    --hidden-dim 256 \
    --optimizer adamw --weight-decay 0.01 --scheduler cosine \
    --head-path parameter_recovery_gru_cosine.pth \
    --log-every 1000 --save-every 5000 \
    2>&1 | tee exp_gru_cosine.log

echo ""
echo "=================================================="
echo "All experiments complete!"
echo "=================================================="
date

# Summary: show all result files
echo ""
echo "=== Results Summary ==="
for f in *_results.json; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        python3 -c "import json; d=json.load(open('$f')); print(f'  Model: {d.get(\"model_type\",\"?\")} | Total Error: {d[\"mean_total_error\"]:.6f} | Improvement: {d[\"improvement_ratio\"]:.2f}x | Best Epoch: {d.get(\"best_epoch\",\"?\")}');"
    fi
done
