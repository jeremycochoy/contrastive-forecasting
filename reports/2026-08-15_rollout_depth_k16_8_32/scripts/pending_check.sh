#!/bin/bash
# #401 — which `pending` markers in the report now have a number.
#
# The report carries its pending cells as the word `pending` in six tables,
# hand written. When the queue scores a cell, somebody has to find every
# marker that the new score fills. Thirteen markers over six tables is where a
# reader finds one the rewrite missed.
#
# This prints each marker with the score file that fills it, so the final pass
# is a list to work through and not a search. It reads, and writes nothing.
#
# Usage:  bash scripts/pending_check.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
REPORT="$CF401_STUDY/$(basename "$CF401_STUDY" | sed 's/^[0-9-]*_//').md"
[ -f "$REPORT" ] || { echo "ABORT: no report at $REPORT" >&2; exit 2; }

n_pending="$(grep -c 'pending' "$REPORT")"
echo "report: $REPORT"
echo "lines holding the word 'pending': $n_pending"
echo

echo "=== cells the queue still owes ==="
ready=0; waiting=0
while read -r tag; do
  [ -n "$tag" ] || continue
  f="$CF401_RESULTS/score_${tag}.txt"
  if [ -s "$f" ]; then
    printf '  READY   %-46s %s\n' "$tag" "$(tr -d ' \n' <"$f")"
    ready=$(( ready + 1 ))
  else
    printf '  waiting %-46s\n' "$tag"
    waiting=$(( waiting + 1 ))
  fi
done <<'TAGS'
k32_bb40k_h40k_student
k32_bb100k_h100k_student
k32_bb200k_h200k_student
k8_bb200k_h200k_student
k32_bb200k_h30k_student_s20260723
k32_bb200k_h30k_student_s20260724
k0_parent_bb100k_h30k_student
TAGS

echo
echo "$ready ready, $waiting waiting"
[ "$waiting" -eq 0 ] && echo "Every cell has scored. The report may drop the word 'pending'."
echo
echo "=== the report's pending lines ==="
grep -n 'pending' "$REPORT" | sed 's/^/  /'
