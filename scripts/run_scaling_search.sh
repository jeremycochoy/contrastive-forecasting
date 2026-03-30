#!/bin/bash
# Scaling search: deeper and wider backbone architectures
# Compare gap trajectories at 200k steps on V2 architecture
set -e

LOGDIR="scaling_search_logs"
mkdir -p "$LOGDIR"

# Auto-select GPU
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU $GPU"

COMMON="--device cuda --encoder-type gru --C 4 --W 32 --ffn-mult 4 --activation gelu --depthwise-conv 3 --dropout 0.1 --lr 7e-5 --dimension 4 --temperature 0.07 --loss-shape cosine_similarity_batch_no_time_neg --T-raw 4096 --val-every 1000 --save-every 50000"

PHASE=${1:-all}

run_experiment() {
    local name=$1 layers=$2 H=$3 nhead=$4 bs=$5
    echo ""
    echo "=========================================="
    echo " $name: ${layers}L H=${H} nhead=${nhead} bs=${bs}"
    echo "=========================================="
    local start=$(date +%s)
    python3 -u train_contrastive_v2.py $COMMON \
        --num-layers $layers --H $H --nhead $nhead --batch-size $bs \
        --total-steps 200000 \
        --save-path "$LOGDIR/scaling_${name}.pth" \
        --experiment-id "scaling_${name}" \
        2>&1 | tee "$LOGDIR/scaling_${name}.log"
    local elapsed=$(( $(date +%s) - start ))
    echo "  >> $name done in ${elapsed}s ($(( elapsed / 60 ))min)"
    sleep 5
}

echo "============================================================"
echo " Scaling Search: 200k steps per config"
echo " Started: $(date)"
echo "============================================================"

START=$(date +%s)

if [[ "$PHASE" == "all" || "$PHASE" == "1" ]]; then
    # Baseline for reference (12L H=1024, same as current best)
    run_experiment "12L_H1024_baseline" 12 1024 8 8
fi

if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
    # Deeper only
    run_experiment "16L_H1024" 16 1024 8 8
    run_experiment "20L_H1024" 20 1024 8 8
fi

if [[ "$PHASE" == "all" || "$PHASE" == "3" ]]; then
    # Wider only
    run_experiment "12L_H1280" 12 1280 10 8
fi

if [[ "$PHASE" == "all" || "$PHASE" == "4" ]]; then
    # Deeper + wider
    run_experiment "16L_H1280" 16 1280 10 8
    run_experiment "20L_H1280" 20 1280 10 6
fi

TOTAL=$(( $(date +%s) - START ))
echo ""
echo "============================================================"
echo " Scaling Search Complete!"
echo " Total time: ${TOTAL}s / $(( TOTAL / 60 ))min"
echo " Finished: $(date)"
echo "============================================================"

echo ""
echo "=== Results Summary ==="
for f in "$LOGDIR"/scaling_*.log; do
    name=$(basename "$f" .log | sed 's/scaling_//')
    peak_gap=$(awk -F'gap=' '{print $2}' "$f" | awk '{print $1}' | sort -rn | head -1)
    final_gap=$(grep "gap=" "$f" | tail -1 | grep -oP 'gap=\K[0-9.]+')
    speed=$(grep "step/s" "$f" | tail -1 | grep -oP '[0-9.]+ step/s')
    echo "  $name: peak_gap=$peak_gap final_gap=$final_gap $speed"
done
