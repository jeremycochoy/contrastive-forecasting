#!/bin/bash
# #390 — bring the 20 cells this experiment did NOT retrain into the new
# report directory, so one results/ tree covers all 30 cells of the #379
# grid.
#
# arm1, arm3, arm4 and bimoco carry no L_align term, so the teacher-target
# change of #390 cannot move them. Their measurements are #379's, copied
# verbatim; the seasonal-naive denominator is byte-identical between the two
# reports, so the 30-cell comparison is on one scale.
#
# arm5 and arm6_v2 are deliberately NOT copied: #390 retrained those ten
# cells and their cell names collide. The pre-teacher arm5 / arm6_v2 numbers
# stay readable at reports/2026-07-21_split_pred_rep_small/results/.
#
#   REPO=/tmp/contrastive-forecasting-390 bash merge_379_cells.sh
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SRC="$REPO/reports/2026-07-21_split_pred_rep_small/results"
DST="${DST:-$REPO/reports/2026-08-04_lalign_teacher/results}"
KEEP='^(arm1|arm3|arm4|bimoco)_'          # cell names
KEEP_RUN='^bb_small_(arm1|arm3|arm4|bimoco)_'   # run names

say(){ echo "[merge379] $*"; }
[ -d "$SRC" ] || { say "source not found: $SRC"; exit 1; }
mkdir -p "$DST"/{eval_gm_mase,training_curves,attn_amplitude}

n_cells=0
for f in "$SRC/eval_gm_mase"/*_summary.txt; do
  [ -e "$f" ] || continue
  cell="$(basename "$f" _summary.txt)"
  echo "$cell" | grep -qE "$KEEP" || continue
  if [ -e "$DST/eval_gm_mase/${cell}_summary.txt" ]; then
    say "REFUSE $cell — already present, a retrained cell must not be overwritten"
    continue
  fi
  csv="$SRC/eval_gm_mase/$cell/all_results.csv"
  rows=0; [ -f "$csv" ] && rows=$(( $(wc -l < "$csv") - 1 ))
  if [ "$rows" -ne 97 ]; then
    say "SKIP $cell — $rows/97 configs in the #379 copy"
    continue
  fi
  cp -f "$f" "$DST/eval_gm_mase/${cell}_summary.txt"
  mkdir -p "$DST/eval_gm_mase/$cell"
  cp -f "$csv" "$DST/eval_gm_mase/$cell/all_results.csv"
  [ -f "$SRC/eval_gm_mase/$cell/summary.txt" ] &&
    cp -f "$SRC/eval_gm_mase/$cell/summary.txt" "$DST/eval_gm_mase/$cell/summary.txt"
  n_cells=$((n_cells + 1))
done
say "cells copied from #379: $n_cells"

for sub in training_curves attn_amplitude; do
  n=0
  for f in "$SRC/$sub"/*.csv; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    echo "$b" | grep -qE "$KEEP_RUN" || continue
    [ -e "$DST/$sub/$b" ] && { say "REFUSE $sub/$b — already present"; continue; }
    cp -f "$f" "$DST/$sub/$b"
    n=$((n + 1))
  done
  say "$sub copied from #379: $n"
done
say "done -> $DST"
