#!/bin/bash
# #373 — run every check that reads the study's raw artefacts, in one place.
#
# The gap-close sessions checked the two new runs against the branch. This goes
# under that: it re-derives the numbers themselves from the evals, the loss
# CSVs and the head logs, so a claim that drifted from its own artefact fails
# here rather than in review.
#
#   verify_scores       99 score files, each recomputed from its own 97-config eval
#   verify_coverage     the 14 x 3 x 2 grid, rebuilt from the score files alone
#   verify_alignx4      item 3's x4 weight and depth 0, read off the loss CSVs
#   verify_provenance   the training machine of every head, read off its own log
#   verify_denominator  the seasonal-naive column, across evals rather than within one
#
# Each check writes its own log under results/ and returns non-zero on failure.
# The script runs all four before it exits, so one failure does not hide three.
#
# Usage: bash verify_close.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
PY="${PY:-/home/jupyter/rnd/.venv/bin/python3}"

rc_all=0
run(){
  local name="$1"; shift
  echo "=== $name"
  "$PY" "$HERE/$name.py" "$@" 2>&1 | tee "$RES/$name.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "  -> $name FAILED rc=$rc"; rc_all=1; }
  echo
}

run verify_scores     --results "$RES"
run verify_coverage   --results "$RES"
run verify_alignx4
run verify_provenance --results "$RES" --tsv "$RES/provenance.tsv"
run verify_denominator --results "$RES"

if [ "$rc_all" -eq 0 ]; then
  echo "ALL CHECKS PASS"
else
  echo "AT LEAST ONE CHECK FAILED" >&2
fi
exit "$rc_all"
