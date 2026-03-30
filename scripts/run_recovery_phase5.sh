#!/bin/bash
# Phase 5: Full 20k-epoch training of best recovery head configurations
# on both V2 (gap=0.203) and V1 (gap=0.105) backbones.
set -e

LOGDIR="recovery_search_logs"
mkdir -p "$LOGDIR"

# Auto-select GPU with most free memory
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU $GPU"

V2_BACKBONE="--model-path v2_2M_model_best.pth --encoder-type gru --H 1024 --num-layers 12 --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3"
V1_MODEL="trained_simple_model_H1024.pth"

run_v2() {
    local name=$1; shift
    echo ""
    echo "=========================================="
    echo " Phase 5: $name (V2 backbone, 20k epochs)"
    echo "=========================================="
    local start=$(date +%s)
    python3 -u train_parameter_recovery_v2.py --device cuda $V2_BACKBONE \
        "$@" --num-arma-params 4 --dimension 4 --epochs 20000 --batch-size 32 \
        --log-every 100 --save-every 5000 \
        --head-path "$LOGDIR/recovery_p5_${name}.pth" \
        2>&1 | tee "$LOGDIR/recovery_p5_${name}.log"
    local elapsed=$(( $(date +%s) - start ))
    echo "  >> $name done in ${elapsed}s"
}

run_v1() {
    local name=$1; shift
    echo ""
    echo "=========================================="
    echo " Phase 5: $name (V1 backbone, 20k epochs)"
    echo "=========================================="
    local start=$(date +%s)
    python3 -u train_parameter_recovery.py --device cuda \
        --model-path "$V1_MODEL" --H 1024 --num-layers 12 \
        "$@" --num-arma-params 4 --dimension 4 --epochs 20000 --batch-size 32 \
        --log-every 100 --save-every 5000 \
        --head-path "$LOGDIR/recovery_p5_${name}.pth" \
        2>&1 | tee "$LOGDIR/recovery_p5_${name}.log"
    local elapsed=$(( $(date +%s) - start ))
    echo "  >> $name done in ${elapsed}s"
}

echo "============================================================"
echo " Phase 5: Full Training (20k epochs)"
echo " Started: $(date)"
echo "============================================================"

START=$(date +%s)

# 1. Best overall: GRU h128 l3 on V2
run_v2 "v2_gru_h128_l3" --model-type gru --hidden-dim 128 --num-gru-layers 3 --loss-type mse
sleep 5

# 2. GRU-pool on V2 (Phase 1 winner)
run_v2 "v2_grupool" --model-type grupool --hidden-dim 256 --loss-type mse
sleep 5

# 3. GRU h128 l3 + Huber on V2
run_v2 "v2_gru_h128_l3_huber" --model-type gru --hidden-dim 128 --num-gru-layers 3 --loss-type huber
sleep 5

# 4. GRU h128 l3 on V1
run_v1 "v1_gru_h128_l3" --model-type gru --hidden-dim 128
sleep 5

# 5. GRU-pool on V1
run_v1 "v1_grupool" --model-type grupool --hidden-dim 256
sleep 5

TOTAL=$(( $(date +%s) - START ))
echo ""
echo "============================================================"
echo " Phase 5 Complete!"
echo " Total time: ${TOTAL}s / $(( TOTAL / 60 ))min"
echo " Finished: $(date)"
echo "============================================================"

echo ""
echo "=== Results Summary ==="
for f in "$LOGDIR"/recovery_p5_*.log; do
    name=$(basename "$f" .log | sed 's/recovery_p5_//')
    imp=$(grep "Improvement" "$f" | tail -1)
    err=$(grep "Mean Total Error" "$f" | tail -1)
    sign_ar=$(grep "Sign Agreement AR" "$f" | tail -1)
    echo "  $name: $err | $imp"
done
