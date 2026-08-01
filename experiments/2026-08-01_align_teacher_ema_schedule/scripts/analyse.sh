#!/bin/bash
# #388 — everything between "the runs finished" and "the report is current".
#
# Usage: WT=/tmp/contrastive-forecasting-388 analyse.sh [gpu]
#
#   1. post-hoc drift over every 5000-step checkpoint of both experiments
#   2. reduce the four #388 training CSVs to the small report tables
#   3. the early / late / slope table
#   4. the six figures
set -euo pipefail

GPU="${1:-0}"
WT="${WT:?WT (worktree root) must be set}"
OUT="$WT/experiments/2026-08-01_align_teacher_ema_schedule"
RUNS_388="${RUNS_388:-/home/jupyter/checkpoints_backup/cf-388}"
RUNS_382="${RUNS_382:-/home/jupyter/checkpoints_backup/cf-382/runs_vast}"

export PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES="$GPU"

python3 "$OUT/scripts/teacher_latent_drift.py" \
    --runs-382 "$RUNS_382" --runs-388 "$RUNS_388" --out-dir "$OUT/results"

python3 "$OUT/scripts/make_results_csvs.py" \
    --runs-388 "$RUNS_388" \
    --artifacts-382 "$WT/experiments/2026-07-28_loss_term_isolation/artifacts" \
    --out-dir "$OUT/results" --stride 100

python3 "$OUT/scripts/make_summary.py" --exp-dir "$OUT"
python3 "$OUT/scripts/make_plots.py" --exp-dir "$OUT"
echo "analysis done"
