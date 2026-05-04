#!/bin/bash
# Full pipeline: backbone training → recovery heads → plots.
# Run from the repo root (or anywhere — script cd's to repo root).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/contrastive-correlation"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# Auto-pick freest GPU if not already set.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
    export CUDA_VISIBLE_DEVICES=$GPU
fi
echo "Using GPU $CUDA_VISIBLE_DEVICES"

EXPID="${EXPID:-corrV1}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
RECOVERY_EPOCHS="${RECOVERY_EPOCHS:-10000}"
SKIP_BACKBONE="${SKIP_BACKBONE:-0}"

CHECKPOINT="$EXP_DIR/checkpoints/${EXPID}.pth"
BACKBONE_RESULTS="$REPO_ROOT/corr_backbone_${EXPID}_results.json"
LOGFILE="$EXP_DIR/logs/${EXPID}.log"

mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/figures"

if [ "$SKIP_BACKBONE" != "1" ]; then
    echo ""
    echo "=========================================="
    echo " Stage 1: contrastive backbone ($EXPID)"
    echo "=========================================="
    python -u experiments/contrastive-correlation/train_contrastive_corr.py \
        --device cuda \
        --encoder-type gru --H 1024 --num-layers 12 \
        --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
        --batch-size 16 --lr 7e-5 \
        --total-steps "$TOTAL_STEPS" --val-every 500 --save-every 5000 \
        --save-path "$CHECKPOINT" \
        --experiment-id "$EXPID" \
        2>&1 | tee "$LOGFILE"
fi

# Pick best-by-gap if available, else final.
BEST_CKPT="${CHECKPOINT%.pth}_best_gap.pth"
[ -f "$BEST_CKPT" ] || BEST_CKPT="$CHECKPOINT"
echo "Using backbone: $BEST_CKPT"

# Train both linear and MLP heads.
for HEAD_TYPE in linear mlp; do
    echo ""
    echo "=========================================="
    echo " Stage 2.$HEAD_TYPE: recovery head"
    echo "=========================================="
    HEAD_PATH="$EXP_DIR/checkpoints/${EXPID}_head_${HEAD_TYPE}.pth"
    python -u experiments/contrastive-correlation/correlation_recovery.py \
        --device cuda \
        --model-path "$BEST_CKPT" \
        --encoder-type gru --H 1024 --num-layers 12 \
        --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
        --head-type "$HEAD_TYPE" --hidden-dim 512 \
        --epochs "$RECOVERY_EPOCHS" --batch-size 32 --lr 3e-4 \
        --head-path "$HEAD_PATH" \
        2>&1 | tee -a "$LOGFILE"
done

# Use the linear head as the canonical head for plots ("single linear" was
# the headline ask). MLP results are in JSON for comparison.
HEAD_TYPE=linear
HEAD_PATH="$EXP_DIR/checkpoints/${EXPID}_head_${HEAD_TYPE}.pth"
HEAD_RESULTS="${HEAD_PATH%.pth}_results.json"
BEST_HEAD="${HEAD_PATH%.pth}_best.pth"
[ -f "$BEST_HEAD" ] || BEST_HEAD="$HEAD_PATH"

echo ""
echo "=========================================="
echo " Stage 3: evaluation + plots"
echo "=========================================="
python -u experiments/contrastive-correlation/evaluate_and_plot.py \
    --device cuda \
    --model-path "$BEST_CKPT" \
    --head-path "$BEST_HEAD" \
    --head-type "$HEAD_TYPE" --hidden-dim 512 \
    --encoder-type gru --H 1024 --num-layers 12 \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --num-samples 400 \
    --out-dir "$EXP_DIR/figures" \
    --backbone-results "$BACKBONE_RESULTS" \
    --head-results "$HEAD_RESULTS" \
    2>&1 | tee -a "$LOGFILE"

echo ""
echo "Done. Figures: $EXP_DIR/figures/"
