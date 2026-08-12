#!/bin/bash
# #373 round 2 — bring each cell's evidence out of its sync tree and into
# the git checkout.
#
# Round 1 ran on elisa and on three boxes, and `collect.sh` reads its
# layout: one directory per box under ~/cf373_sync, plus the durable root.
# Round 2 is one directory per CELL under ~/cf373_r2, because a cell owns a
# box for its whole ladder. This script reads that layout.
#
# What comes across, and what does not:
#
#   gift/all_results.csv   the 97 per-config numbers. Every figure that
#   gift/summary.txt       splits by horizon or by domain reads these, and
#                          the aggregate score is read off the summary.
#   eval_local.log         which backbone and which head produced the score.
#   worker/head/run logs   what ran, when, and on which card.
#   losses CSV             downsampled — the raw file is 20 to 31 MB per
#                          leg and there are 27 legs.
#
# Checkpoints stay out: 5 MB each, 200-odd of them, and they are not
# evidence a reader of the report needs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="${1:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
SYNC_BASE="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"
CELLS="A1 A2 A3 A4 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10"
mkdir -p "$DST/results/eval" "$DST/results/r2_logs" "$DST/curves/r2"

say(){ echo "[r2-collect] $*"; }
n_eval=0 n_log=0 n_curve=0

for c in $CELLS; do
  base="$SYNC_BASE/$c"
  [ -d "$base" ] || continue

  # ---- the evals, one directory per (cell, stop, head) --------------------
  for d in "$base"/sync/eval/*/; do
    [ -f "$d/gift/all_results.csv" ] || continue
    tag="$(basename "$d")"
    rows=$(( $(wc -l <"$d/gift/all_results.csv") - 1 ))
    # A short CSV is a partial eval. Copying it would let a figure average
    # over fewer than 97 configs and read as if it were the study's metric.
    [ "$rows" -eq 97 ] || { say "skip $tag: $rows configs, want 97"; continue; }
    mkdir -p "$DST/results/eval/$tag"
    for f in gift/all_results.csv gift/summary.txt eval_local.log backbone.txt; do
      [ -f "$d/$f" ] && cp -f "$d/$f" "$DST/results/eval/$tag/$(basename "$f")"
    done
    n_eval=$(( n_eval + 1 ))
  done

  # ---- the logs -----------------------------------------------------------
  for f in "$base"/results/*.log; do
    [ -f "$f" ] || continue
    case "$f" in *.prev) continue;; esac
    cp -f "$f" "$DST/results/r2_logs/${c}_$(basename "$f")" && n_log=$(( n_log + 1 ))
  done

  # ---- the training curves, downsampled ----------------------------------
  # The header and every 200th row: a 200-step grid, finer than the 500-step
  # grid the report's curves are drawn on, at 1/200th of 31 MB.
  # `losses_csv.py` is a library the plot scripts import, not a filter, so
  # the thinning is here.
  for f in $(find "$base/sync" -name "*_losses.csv" ! -name "*.prev" 2>/dev/null); do
    out="$DST/curves/r2/${c}_$(basename "$f")"
    awk 'NR==1 || NR%200==2' "$f" >"$out" && n_curve=$(( n_curve + 1 ))
  done
done

say "evals $n_eval, logs $n_log, curves $n_curve -> $DST"
