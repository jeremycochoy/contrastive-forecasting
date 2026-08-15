#!/bin/bash
# #373 round 3 — bring the flat round-3 tree's evidence into the git checkout.
#
# Round 2 kept one directory per CELL under ~/cf373_r2, and `r2_collect.sh`
# reads that layout. Round 3 keeps ONE flat root, ~/cf373_r3, mirroring the
# box's /root/cf373_runs, so one sync loop fills it and `cell_paths.sh`
# resolves a checkpoint the same way on both machines. This script reads
# that layout.
#
# Without it the 200k numbers never reach a figure: every split and plot
# script reads `results/eval/*/all_results.csv` in the checkout, and round
# 3's evals land in `$CF373_R3/eval/<tag>/gift/`.
#
# What comes across, and what does not:
#
#   gift/all_results.csv   the 97 per-config numbers. Every figure that
#   gift/summary.txt       splits by horizon or by domain reads these, and
#                          the aggregate score is read off the summary.
#   eval_local.log         which backbone and which head produced the score.
#   backbone.txt           the checkpoint the head read, so a pair that is
#                          not this cell's cannot be scored as if it were.
#   head.log               the head's own steps, seed and card.
#   run logs               what trained, when, and on which card.
#   losses CSV             downsampled — the raw file is 10 to 31 MB per leg.
#
# Checkpoints stay out: 5 MB each, and they are not evidence a reader of the
# report needs.
#
# Idempotent: run it as often as you like, including while the queue runs.
# A partial eval is skipped, not copied, so no figure can average over fewer
# than 97 configs and read as if it were the study's metric.
#
# Usage: bash r3_collect.sh [git checkout root]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="${1:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}"
R3_LOGS="$(dirname "$R3")/results"
mkdir -p "$DST/results/eval" "$DST/results/r3_logs" "$DST/curves/r3"

say(){ echo "[r3-collect] $*"; }
n_eval=0 n_log=0 n_curve=0 n_skip=0

[ -d "$R3" ] || { say "no round-3 root at $R3"; exit 0; }

# ---- the evals, one directory per (cell, stop, head) ----------------------
for d in "$R3"/eval/*/; do
  [ -f "$d/gift/all_results.csv" ] || continue
  tag="$(basename "$d")"
  rows=$(( $(wc -l <"$d/gift/all_results.csv") - 1 ))
  if [ "$rows" -ne 97 ]; then
    say "skip $tag: $rows configs, want 97"; n_skip=$(( n_skip + 1 )); continue
  fi
  mkdir -p "$DST/results/eval/$tag"
  for f in gift/all_results.csv gift/summary.txt eval_local.log backbone.txt \
           head.log; do
    [ -f "$d/$f" ] && cp -f "$d/$f" "$DST/results/eval/$tag/$(basename "$f")"
  done
  n_eval=$(( n_eval + 1 ))
done

# ---- the trainer logs the sync loop pulled off the box --------------------
if [ -d "$R3_LOGS" ]; then
  for f in "$R3_LOGS"/*.log; do
    [ -f "$f" ] || continue
    case "$f" in *.prev) continue;; esac
    cp -f "$f" "$DST/results/r3_logs/$(basename "$f")" && n_log=$(( n_log + 1 ))
  done
fi

# ---- the training curves, downsampled ------------------------------------
# The header and every 200th row: a 200-step grid, finer than the 500-step
# grid the report's curves are drawn on, at 1/200th of the raw file. The run
# directory name carries the cell's recipe, so the copy keeps it verbatim
# rather than re-deriving a cell id that `cell_paths.sh` already owns.
while IFS= read -r f; do
  [ -n "$f" ] || continue
  rel="${f#$R3/}"
  awk 'NR==1 || NR%200==2' "$f" >"$DST/curves/r3/${rel//\//__}" \
    && n_curve=$(( n_curve + 1 ))
done < <(find "$R3" -name "*_losses.csv" ! -name "*.prev" 2>/dev/null)

say "evals $n_eval (skipped $n_skip partial), logs $n_log, curves $n_curve -> $DST"
