#!/bin/bash
# #373 — give every cell score the one name the scripts read.
#
# A score file is named `score_<CELL>_k3_bb<STOP>k_<HEAD>.txt`. One pair did
# not follow it: B1's bb40k numbers were written in round 1, under the probe
# tag `G6_B1_k3_bb40k_*`, before round 2 named the cells. Every script that
# globs the cell pattern therefore missed two real numbers, and the coverage
# table needed a hand-written alias to see them.
#
# The rename is safe because the round-1 eval read B1's own checkpoint. Its
# log names `..._cf373k3_40k.pth` under the backup root, and that file is
# md5-identical to the copy round 2 resumed:
#   23ba3d9dcb4a9ee86d18a377a5965ff1
#
# The probe-tagged copies stay where they are. This adds the canonical name
# beside them; it removes nothing.
#
# Usage: bash normalise_scores.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(dirname "$HERE")/results"

# from-tag  to-tag
MAP="
G6_B1_k3_bb40k_student	B1_k3_bb40k_student
G6_B1_k3_bb40k_teacher	B1_k3_bb40k_teacher
"

n=0
while read -r from to; do
  [ -n "$from" ] || continue
  src="$RES/score_$from.txt"
  dst="$RES/score_$to.txt"
  [ -s "$src" ] || { echo "SKIP $from — no score file"; continue; }
  if [ -s "$dst" ]; then
    if [ "$(cat "$src")" = "$(cat "$dst")" ]; then
      echo "OK   $to already holds $(cat "$dst")"
    else
      echo "ABORT: $dst holds $(cat "$dst"), $src holds $(cat "$src")" >&2
      exit 1
    fi
  else
    cp -f "$src" "$dst"
    echo "WROTE score_$to.txt = $(cat "$dst")   (from $from)"
    n=$(( n + 1 ))
  fi
  # The eval artefacts move with the name, so a reader who follows the score
  # to its 97 rows lands in a directory named for the cell.
  if [ -d "$RES/eval/$from" ] && [ ! -d "$RES/eval/$to" ]; then
    cp -r "$RES/eval/$from" "$RES/eval/$to"
    printf '%s\n' "$from" > "$RES/eval/$to/renamed_from.txt"
    echo "     eval/$to <- eval/$from"
  fi
done <<< "$MAP"

echo "normalised $n score name(s)"
