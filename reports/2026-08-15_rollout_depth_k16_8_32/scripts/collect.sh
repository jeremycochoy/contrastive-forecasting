#!/bin/bash
# #401 — every score file this study wrote, in one CSV.
#
# `head_eval_bb.sh` writes one `score_<tag>.txt` per (depth, stop, head
# budget). The tag carries all three, so the table is read back out of the
# filenames rather than kept in a second place that can drift from them.
#
# The phase is derived, not stored: a head budget equal to the backbone stop
# is phase 2, anything else is phase 1. That is the card's own definition of
# the two phases.
#
# An empty score file is skipped, not read as 0. An eval killed between
# opening and writing leaves one, and a 0.0 in this CSV would be the best
# GM-Relative MASE the project ever recorded.
#
# Usage:  bash collect.sh            # writes results/scores.csv
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="$CF401_RESULTS/scores.csv"
mkdir -p "$CF401_RESULTS"

{
  echo "phase,k,stop,head_steps,encoder,score"
  for f in "$CF401_RESULTS"/score_k*.txt; do
    [ -e "$f" ] || continue
    [ -s "$f" ] || continue
    score="$(tr -d ' \t\r\n' <"$f")"
    [ -n "$score" ] || continue
    # score_k<K>_bb<N>k_h<M>k_<enc>.txt
    base="$(basename "$f" .txt)"
    fields="$(printf '%s\n' "${base#score_}" \
      | sed -nE 's/^k([0-9]+)_bb([0-9]+)k_h([0-9]+)k_(.+)$/\1 \2 \3 \4/p')"
    [ -n "$fields" ] || { echo "WARN: unparsed score file $base" >&2; continue; }
    read -r k stop_k head_k enc <<<"$fields"
    stop=$(( stop_k * 1000 )); head=$(( head_k * 1000 ))
    phase=1; [ "$head" -eq "$stop" ] && phase=2
    echo "$phase,$k,$stop,$head,$enc,$score"
  done
} >"$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"
echo "$OUT: $(( $(wc -l <"$OUT") - 1 )) score(s)"
