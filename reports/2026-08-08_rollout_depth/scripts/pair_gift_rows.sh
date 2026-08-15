#!/bin/bash
# #373 — do two cells' 97-config eval CSVs hold the same rows?
#
# `pair_identity.py` compares WEIGHTS and `pair_head_files.py` compares
# FILES. Neither reads the eval. The card blocked on a third question: does
# the eval path key ignore the EMA regime, so that two cells write one set
# of numbers? Only the per-config CSVs answer it.
#
# A1 and B3 are the pair the card named. Each cell-stop-head has its own
# `gift/all_results.csv` under its own cell-id directory. This script diffs
# them row by row, sorted, so row order cannot fake a match.
#
# Reading: 0 differing lines means the two runs agree on every one of the 97
# configs. 194 means all 97 differ (one `<` and one `>` line per config).
#
# Usage: bash pair_gift_rows.sh [out.tsv]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
OUT="${1:-$STUDY/results/pair_A1B3_gift_rows.tsv}"
R2="${CF373_R2:-/home/jupyter/cf373_r2}"

{
  printf 'stop_k\tenc\trows_A1\trows_B3\tdiffering_lines\tA1_csv\tB3_csv\n'
  for stop in 40k 100k; do
    for enc in student teacher; do
      a="$R2/A1/sync/eval/A1_k3_bb${stop}_${enc}/gift/all_results.csv"
      b="$R2/B3/sync/eval/B3_k3_bb${stop}_${enc}/gift/all_results.csv"
      if [ -f "$a" ] && [ -f "$b" ]; then
        n=$(diff <(sort "$a") <(sort "$b") | grep -c '^[<>]')
        printf '%s\t%s\t%d\t%d\t%d\t%s\t%s\n' \
          "${stop%k}000" "$enc" \
          "$(( $(wc -l < "$a") - 1 ))" "$(( $(wc -l < "$b") - 1 ))" "$n" "$a" "$b"
      else
        printf '%s\t%s\t-\t-\t-\t%s\t%s\n' "${stop%k}000" "$enc" "$a" "$b"
      fi
    done
  done
} > "$OUT"

cat "$OUT"
