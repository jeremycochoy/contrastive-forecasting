#!/bin/bash
# End-to-end pipeline that reproduces the V4 numbers in contrastive-correlation.md:
# backbone (200k steps, bs=16, cosine LR) → TimeAware recovery head
# (30k epochs, bs=8, cosine LR, latent=h) → figures.
#
# Run from the repo root, or anywhere — script cd's to repo root.
# Knobs (env vars):
#   EXPID            checkpoint/log basename (default corrV4)
#   TOTAL_STEPS      backbone steps (default 200000)
#   RECOVERY_EPOCHS  recovery-head epochs (default 30000)
#   SKIP_BACKBONE=1  skip stage 1 if you already have a backbone checkpoint
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/2026-05-04_contrastive-correlation"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
    export CUDA_VISIBLE_DEVICES=$GPU
fi
echo "Using GPU $CUDA_VISIBLE_DEVICES"

EXPID="${EXPID:-corrV4}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
RECOVERY_EPOCHS="${RECOVERY_EPOCHS:-30000}"
SKIP_BACKBONE="${SKIP_BACKBONE:-0}"

CHECKPOINT="$EXP_DIR/checkpoints/${EXPID}.pth"
LOGFILE="$EXP_DIR/logs/${EXPID}.log"
HEAD_PATH="$EXP_DIR/checkpoints/${EXPID}_head_timeaware_h.pth"
HEAD_RESULTS="${HEAD_PATH%.pth}_results.json"

mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/logs" "$EXP_DIR/plots"

if [ "$SKIP_BACKBONE" != "1" ]; then
    echo ""
    echo "=========================================="
    echo " Stage 1: contrastive backbone ($EXPID)"
    echo "=========================================="
    python -u experiments/2026-05-04_contrastive-correlation/scripts/train_contrastive_corr.py \
        --device cuda \
        --encoder-type gru --H 1024 --num-layers 12 \
        --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
        --batch-size 16 --lr 7e-5 \
        --total-steps "$TOTAL_STEPS" --val-every 1000 --save-every 5000 \
        --lr-schedule cosine --warmup-steps 0 --grad-clip 0 \
        --save-path "$CHECKPOINT" \
        --experiment-id "$EXPID" \
        2>&1 | tee "$LOGFILE"
fi

# Use the best-by-gap checkpoint if it exists.
BEST_CKPT="${CHECKPOINT%.pth}_best_gap.pth"
[ -f "$BEST_CKPT" ] || BEST_CKPT="$CHECKPOINT"
echo "Using backbone: $BEST_CKPT"
BACKBONE_RESULTS="$EXP_DIR/checkpoints/corr_backbone_${EXPID}_results.json"
[ -f "$BACKBONE_RESULTS" ] || BACKBONE_RESULTS="$REPO_ROOT/corr_backbone_${EXPID}_results.json"

echo ""
echo "=========================================="
echo " Stage 2: TimeAware recovery head"
echo "=========================================="
python -u experiments/2026-05-04_contrastive-correlation/scripts/correlation_recovery.py \
    --device cuda \
    --model-path "$BEST_CKPT" \
    --encoder-type gru --H 1024 --num-layers 12 \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --head-type timeaware --hidden-dim 512 --latent h \
    --epochs "$RECOVERY_EPOCHS" --batch-size 8 --lr 3e-4 \
    --lr-schedule cosine --warmup-epochs 0 --grad-clip 0 \
    --head-path "$HEAD_PATH" \
    2>&1 | tee -a "$LOGFILE"

BEST_HEAD="${HEAD_PATH%.pth}_best.pth"
[ -f "$BEST_HEAD" ] || BEST_HEAD="$HEAD_PATH"

echo ""
echo "=========================================="
echo " Stage 3: evaluation + plots"
echo "=========================================="
python -u experiments/2026-05-04_contrastive-correlation/scripts/evaluate_and_plot.py \
    --device cuda \
    --model-path "$BEST_CKPT" \
    --head-path "$BEST_HEAD" \
    --head-type timeaware --hidden-dim 512 --latent h \
    --encoder-type gru --H 1024 --num-layers 12 \
    --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
    --num-samples 800 \
    --out-dir "$EXP_DIR/plots" \
    --backbone-results "$BACKBONE_RESULTS" \
    --head-results "$HEAD_RESULTS" \
    2>&1 | tee -a "$LOGFILE"

echo ""
echo "Done. Figures: $EXP_DIR/plots/"
