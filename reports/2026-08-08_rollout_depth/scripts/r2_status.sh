#!/bin/bash
# #373 round 2 — one screen of where the study is.
#
# Usage: bash r2_status.sh
#
# Per cell: the box, the furthest backbone checkpoint in the LOCAL sync tree
# (not the box's — a step that has not been pulled is a step that can be
# lost), the heads that exist, and the scores that are in.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
SYNC_BASE="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"
BOXES="$RES/r2_boxes.tsv"
CELLS="A1 A2 A3 A4 B1 B2 B3 B4 B5 B6 B7 B8 B9 B10"

printf '%-4s %-11s %-9s %-24s %-11s %s\n' CELL BOX STATE "BB CKPTS (local)" HEADS SCORES
for c in $CELLS; do
  row="$(grep -P "^$c\t" "$BOXES" 2>/dev/null | head -1)"
  id=$(cut -f2 <<<"$row"); host=$(cut -f3 <<<"$row")
  box="${id:--}"
  state="-"
  [ -n "$row" ] && state="up"
  [ -f "$RES/r2_reaped_$c" ] && state="reaped"

  steps=$(find "$SYNC_BASE/$c/sync" -name "*_[0-9]*k.pth" ! -name "*optimizer*" 2>/dev/null \
          | sed -E 's/.*_([0-9]+)k\.pth$/\1/' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')
  heads=$(ls -d "$SYNC_BASE/$c"/sync/eval/*/ 2>/dev/null | while read -r d; do
            ls "$d"/qhead_*_final.pth >/dev/null 2>&1 && basename "$d"; done \
          | sed -E "s/^${c}_k[0-9]+_bb//;s/k_student/S/;s/k_teacher/T/" | tr '\n' ',' | sed 's/,$//')
  scores=$(ls "$RES"/score_${c}_k*_bb*.txt 2>/dev/null \
           | sed -E "s|.*/score_${c}_k[0-9]+_bb||;s/k_student\.txt/S/;s/k_teacher\.txt/T/" \
           | tr '\n' ',' | sed 's/,$//')
  printf '%-4s %-11s %-9s %-24s %-11s %s\n' "$c" "$box" "$state" "${steps:--}" "${heads:--}" "${scores:--}"
done

echo
echo "-- vast.ai --"
(cd "$(cd "$HERE" && git rev-parse --show-toplevel)" && vastrun-status 2>&1 | tail -20)
(cd "$(cd "$HERE" && git rev-parse --show-toplevel)" && vastrun-balance 2>&1 | head -1)
