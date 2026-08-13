#!/bin/bash
# #373 — one command from "both scores are on disk" to "the comment is ready".
#
# Order matters and each step names why it runs where it does:
#
#   1 collect      the eval CSVs live under CF373_ROOT, and every rebuild
#                  step reads them from the CHECKOUT. A rebuild before this
#                  silently skips the run that just finished.
#   2 rebuild      splits, the paired bootstraps (the four new B1 labels
#                  among them), the figures, the coverage-free tables, and
#                  the injection into the report.
#   3 verdict      gap3_item3.py, whose rule was committed before the
#                  numbers existed. It refuses to write without all six
#                  scores and all six intervals.
#   4 coverage     the grid by value, by markdown and by stage.
#   5 comment      built, not posted. `--dry` stops before `gh`.
#
# Every step is idempotent, so a re-run after a fix costs only its own time.
#
# Usage: bash scripts/gap3_close.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
GIT_ROOT="${GIT_ROOT:-/tmp/contrastive-forecasting-373}"
RES="$STUDY/results"

step(){ echo; echo "=== $* ==="; }
fail=0

step "1/5 collect"
bash "$HERE/collect.sh" "$GIT_ROOT" 2>&1 | tail -8 || fail=1

step "2/5 rebuild"
bash "$HERE/make_report_assets.sh" "$GIT_ROOT" 2>&1 | tail -45 || fail=1

step "3/5 item-3 verdict"
python3 "$HERE/gap3_item3.py" --results "$RES" 2>&1 || fail=1

step "4/5 coverage"
timeout 300 python3 "$HERE/r2_coverage.py"         >"$RES/coverage.txt" 2>&1
timeout 300 python3 "$HERE/r2_coverage.py" --md    >"$RES/coverage.md" 2>&1
timeout 300 python3 "$HERE/r2_coverage.py" --state >"$RES/coverage_state.txt" 2>&1
grep -m1 '^deliverables' "$RES/coverage.txt" || fail=1

step "5/5 comment, not posted"
bash "$HERE/gap_close_comment.sh" --dry 2>&1 | tail -6 || fail=1

echo
echo "gap3_close: fail=$fail"
exit "$fail"
