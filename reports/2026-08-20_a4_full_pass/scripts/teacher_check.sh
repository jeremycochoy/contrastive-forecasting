#!/bin/bash
# #407 review gap 4 — run the teacher tensor check on every pair on disk.
#
# `teacher_move.py` compares two checkpoints. This finds the pairs. It runs
# from the watchdog, so the answer for a stop lands within half an hour of
# the checkpoint, and it skips a pair it has already written.
#
# The pairs are consecutive stops along the one trajectory: 100k to 200k,
# 200k to 300k, 300k to 450k, 450k to 665k. The first pair is the control
# the report needs, because it straddles nothing: both ends sit past the
# end of the EMA ramp.
#
# 40k to 100k is the OTHER control, and it must show movement. It is not in
# this list because round 3's tree holds no 40k checkpoint; the report reads
# it from round 2's tree, and `results/teacher_move_40k_100k.json` records
# it.
#
# Usage: [CF373_ROOT=<root>] teacher_check.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
RUNS="${CF373_ROOT:-${RUNS:-/home/jupyter/cf373_r3/sync}}"
mkdir -p "$RES"

PAIRS="${PAIRS:-100000:200000 200000:300000 300000:450000 450000:665000}"
n=0 skip=0
for pair in $PAIRS; do
  a="${pair%%:*}"; b="${pair##*:}"
  out="$RES/teacher_move_$(( a / 1000 ))k_$(( b / 1000 ))k.json"
  [ -s "$out" ] && { skip=$(( skip + 1 )); continue; }
  python3 "$HERE/teacher_move.py" --root "$RUNS" --pair "$a" "$b" \
    --json "$out" >"${out%.json}.txt" 2>&1 || {
      # A stop whose checkpoint has not landed yet is the normal case here,
      # not a fault. Take both files away so the next tick tries again and
      # the results directory holds only answers.
      rm -f "$out" "${out%.json}.txt"; continue; }
  echo "[cf407-teacher] $(( a / 1000 ))k -> $(( b / 1000 ))k: $(tail -1 "${out%.json}.txt")"
  n=$(( n + 1 ))
done
echo "[cf407-teacher] wrote $n, already had $skip"
