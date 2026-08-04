#!/bin/bash
# #390 review item 3 — the student control, end to end.
#
# Backbone 0 -> 40k with `--align-target student` on THIS branch, then the
# wave-1 head (15 000 steps) and the full 97-config GIFT-Eval. The question
# it answers: does this branch reproduce #379's arm5 40k number (1.5478)?
# If it does, the 20 teacher-vs-student deltas are within-code. If it does
# not, they are cross-experiment and the report must say so.
#
# Same three stages as a wave, driven in one process so the eval starts the
# moment the backbone lands.
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train BB_GPU=0 bash run_student_control.sh
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
ARM="${ARM:-arm5}"
BB_GPU="${BB_GPU:?set BB_GPU=0 or 1}"
BB_STEP_K="${BB_STEP_K:-40}"
HEAD_STEPS="${HEAD_STEPS:-15000}"

RES="$WT/experiments/2026-08-01_lalign_teacher/results"
mkdir -p "$RES"
LOG="$RES/student_control_${ARM}.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [student-control/$ARM] $*" | tee -a "$LOG"; }

say "stage 1/2 — backbone 0 -> ${BB_STEP_K}k, --align-target student, GPU $BB_GPU"
WT="$WT" BB_GPU="$BB_GPU" \
  TARGET_STEPS=$(( BB_STEP_K * 1000 )) SAVE_EVERY=10000 \
  bash "$HERE/run_arm_student.sh" "$ARM"
rc=$?
say "stage 1/2 rc=$rc"
[ $rc -eq 0 ] || exit $rc

say "stage 2/2 — head ${HEAD_STEPS} steps + GIFT-Eval 97 configs"
CF390_NAME_SUFFIX=alignstudent CELL_TAG=_alignstudent \
  WT="$WT" ARM="$ARM" BB_GPU="$BB_GPU" \
  BB_STEP_K="$BB_STEP_K" HEAD_STEPS="$HEAD_STEPS" \
  bash "$HERE/eval_arm.sh"
rc=$?
say "stage 2/2 rc=$rc"
[ $rc -eq 0 ] || exit $rc

SUM="$WT/experiments/2026-08-01_lalign_teacher/eval_gm_mase/${ARM}_alignstudent_bb${BB_STEP_K}k_hd${HEAD_STEPS}s_summary.txt"
say "DONE — $(cat "$SUM" 2>/dev/null || echo 'no summary')"
