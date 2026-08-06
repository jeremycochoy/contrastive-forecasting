#!/bin/bash
# #393 — show, mechanically, that the four `alignT` cells were trained by the
# same code as the parent report's teacher cells.
#
# Usage:  bash scripts/verify_align_target_port.sh [ref]
#         (default ref: PR #392's head, feature/contrastive-forecasting-390)
#
# WHY THIS EXISTS. `--align-target teacher` while the main contrastive loss
# is on is rejected on `experiments`; the capability sits in open PR #392,
# and #390's teacher runs — the parent report this card's four teacher cells
# are compared with — were trained on that branch. All four cells here died
# at step 0 until the minimal piece was ported across in adde0cc. If the
# port differs from #392 in any way that touches the objective, the four
# cells are not comparable with the parent and the head-to-head this card
# exists to make is not one.
#
# The check is a diff, not a reading:
#
#   1. src/loss.py, whole file, against #392's. Byte-identical or fail.
#   2. every line of train.py mentioning align_target / align_ref /
#      --align-target, against #392's, in order. Identical or fail.
#
# `student` resolving to the same expression either side is what the whole
# file being identical means, so it needs no separate assertion: the six
# student cells run the code they always ran.
#
# Writes results/align_target_port.txt.
set -uo pipefail

REF="${1:-3214dcf15a8d3c481492cade8e05118ec35af412}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="${WT:-$(dirname "$(dirname "$EXP")")}"
OUT="$EXP/results/align_target_port.txt"
TRAIN=experiments/2026-04-27_freq-embedding/scripts/train.py
cd "$WT" || exit 2

fail=0
{
  echo "#393 — the --align-target teacher port, against PR #392"
  echo
  echo "ran            $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "this branch    $(git rev-parse HEAD)  $(git rev-parse --abbrev-ref HEAD)"
  echo "reference      $REF  (PR #392 head, branch feature/contrastive-forecasting-390)"
  echo "port commit    $(git log --format='%h %ci %s' -1 --all --grep='L_align targets the teacher inside the main contrastive loss' | head -1)"
  echo

  echo "1. src/loss.py, whole file"
  if git diff --quiet "$REF" HEAD -- src/loss.py; then
    echo "   IDENTICAL — $(git show HEAD:src/loss.py | wc -l) lines, sha256 $(git show HEAD:src/loss.py | sha256sum | cut -d' ' -f1)"
  else
    echo "   DIFFERS:"
    git diff "$REF" HEAD -- src/loss.py | sed 's/^/   /'
    fail=1
  fi
  echo

  echo "2. train.py, every line naming align_target / align_ref / --align-target"
  a=$(git show "$REF:$TRAIN" | grep -E 'align_target|align_ref|align-target')
  b=$(git show "HEAD:$TRAIN" | grep -E 'align_target|align_ref|align-target')
  if [ "$a" = "$b" ]; then
    echo "   IDENTICAL — $(printf '%s\n' "$b" | grep -c .) line(s):"
    printf '%s\n' "$b" | sed 's/^/     /'
  else
    echo "   DIFFERS:"
    diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") | sed 's/^/   /'
    fail=1
  fi
  echo

  if [ "$fail" -eq 0 ]; then
    echo "VERDICT: the objective this card trained is the objective #390 trained."
    echo "The four alignT cells are comparable with the parent report, and the six"
    echo "student cells resolve to the expression they resolved to before the port."
  else
    echo "VERDICT: FAILED — the port and #392 have diverged. The teacher cells are"
    echo "not comparable with the parent report until this is explained."
  fi
} > "$OUT" 2>&1

cat "$OUT"
exit "$fail"
