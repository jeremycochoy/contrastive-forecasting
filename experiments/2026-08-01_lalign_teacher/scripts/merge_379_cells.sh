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
# arm5 and arm6_v2 ARE copied too, under an `_alignstudent379` cell tag.
# Those ten arms are the ones #390 retrained, so their bare cell names belong
# to the teacher measurement; the tag keeps both on one scale in one
# directory without either overwriting the other. Without them the central
# comparison of this report — teacher against student at a fixed backbone
# step — cannot be rebuilt from `gm_relative_mase.csv` (review item 6).
#
# The `379` in the tag is load-bearing. The student control (review item 3)
# is the same arm and the same target measured HERE, and lands under a plain
# `_alignstudent`; the two must not share a name.
#
#   REPO=/tmp/contrastive-forecasting-390 bash merge_379_cells.sh
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SRC="$REPO/reports/2026-07-21_split_pred_rep_small/results"
DST="${DST:-$REPO/reports/2026-08-04_lalign_teacher/results}"
KEEP='^(arm1|arm3|arm4|bimoco)_'          # cell names, copied as-is
KEEP_STUDENT='^(arm5|arm6_v2)(_(tr1|nse|ncpc|combab))?_bb'  # copied tagged
STUDENT_TAG='_alignstudent379'
KEEP_RUN='^bb_small_(arm1|arm3|arm4|bimoco)_'   # run names

say(){ echo "[merge379] $*"; }
[ -d "$SRC" ] || { say "source not found: $SRC"; exit 1; }
mkdir -p "$DST"/{eval_gm_mase,training_curves,attn_amplitude}

n_cells=0; n_student=0
for f in "$SRC/eval_gm_mase"/*_summary.txt; do
  [ -e "$f" ] || continue
  cell="$(basename "$f" _summary.txt)"
  # An untagged copy for the 20 arms with no L_align term; a tagged one for
  # the 10 arms #390 retrained, whose bare names are the teacher cells.
  if echo "$cell" | grep -qE "$KEEP"; then
    dst_cell="$cell"
  elif echo "$cell" | grep -qE "$KEEP_STUDENT"; then
    # arm5_nse_bb40k_hd15000s -> arm5_nse_alignstudent_bb40k_hd15000s
    dst_cell="$(echo "$cell" | sed -E "s/_bb([0-9]+k_hd[0-9]+s)$/${STUDENT_TAG}_bb\1/")"
    [ "$dst_cell" != "$cell" ] || { say "SKIP $cell — could not tag"; continue; }
  else
    continue
  fi
  if [ -e "$DST/eval_gm_mase/${dst_cell}_summary.txt" ]; then
    say "REFUSE $dst_cell — already present, a retrained cell must not be overwritten"
    continue
  fi
  csv="$SRC/eval_gm_mase/$cell/all_results.csv"
  rows=0; [ -f "$csv" ] && rows=$(( $(wc -l < "$csv") - 1 ))
  if [ "$rows" -ne 97 ]; then
    say "SKIP $cell — $rows/97 configs in the #379 copy"
    continue
  fi
  cp -f "$f" "$DST/eval_gm_mase/${dst_cell}_summary.txt"
  mkdir -p "$DST/eval_gm_mase/$dst_cell"
  cp -f "$csv" "$DST/eval_gm_mase/$dst_cell/all_results.csv"
  [ -f "$SRC/eval_gm_mase/$cell/summary.txt" ] &&
    cp -f "$SRC/eval_gm_mase/$cell/summary.txt" "$DST/eval_gm_mase/$dst_cell/summary.txt"
  if [ "$dst_cell" = "$cell" ]; then n_cells=$((n_cells + 1));
  else n_student=$((n_student + 1)); fi
done
say "cells copied from #379: $n_cells untagged + $n_student tagged $STUDENT_TAG"

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
