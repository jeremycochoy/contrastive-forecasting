#!/bin/bash
# #390 — the cross-code-boundary control.
#
# Every teacher-vs-student delta in this report compares a #390 number
# (this branch) against a #379 number (older code, never re-run). This
# script re-runs one arm on THIS branch with L_align pointed back at the
# student, so at least one cell is measured on both sides of that boundary.
#
# It does not restate run_arm.sh's command line. It DERIVES the launcher
# from run_arm.sh by exactly three textual substitutions:
#
#   align-target teacher  ->  align-target student   (the flag under test)
#   alignteacher          ->  alignstudent           (run name, no collision)
#   dl_${ARM}.log         ->  dl_${ARM}_student.log  (its own launcher log)
#
# so the control's trainer invocation cannot drift from the teacher arms'
# by anything except the flag. `tests/test_390_student_control.py` pins the
# transformation and checks the resulting arm5 flag set against #379's own
# launcher.
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train BB_GPU=0 TARGET_STEPS=40000 \
#     SAVE_EVERY=10000 bash run_arm_student.sh arm5
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/run_arm.sh"
[ -f "$SRC" ] || { echo "ABORT: no run_arm.sh next to $0" >&2; exit 2; }

GEN="${GEN:-$HERE/.run_arm_student.generated.sh}"
sed -e 's/align-target teacher/align-target student/g' \
    -e 's/alignteacher/alignstudent/g' \
    -e 's/dl_\${ARM}\.log/dl_\${ARM}_student.log/g' \
    "$SRC" > "$GEN"

# A substitution that silently matched nothing would run the teacher arm
# under a student-looking name. Refuse rather than mislabel a checkpoint.
grep -q -- '--align-target student' "$GEN" || {
  echo "ABORT: generated launcher has no --align-target student" >&2; exit 3; }
grep -q -- 'align-target teacher' "$GEN" && {
  echo "ABORT: generated launcher still mentions align-target teacher" >&2; exit 3; }
grep -q -- 'alignteacher' "$GEN" && {
  echo "ABORT: generated launcher still carries the alignteacher name" >&2; exit 3; }
grep -q -- 'dl_${ARM}_student.log' "$GEN" || {
  echo "ABORT: generated launcher did not get its own launcher log" >&2; exit 3; }

exec bash "$GEN" "$@"
