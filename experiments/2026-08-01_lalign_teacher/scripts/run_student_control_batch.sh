#!/bin/bash
# #390 review item 1 (second pass) — the same-branch student control on the
# OTHER NINE cells, at backbone step 40 000.
#
# Why this exists. The arm5 control came back at 1.4501 where #379's arm5
# 40k cell reads 1.5478 — the same arm, the same `--align-target student`,
# the same seed 20260520, the same command line, 0.0977 apart. That gap is
# the code snapshot, not the flag. Every teacher-vs-#379 delta in the wave
# tables carries it. The only way to say what the flag is worth is to hold
# the snapshot fixed, so this runs the student arm of all ten cells on THIS
# branch and stops at 40k.
#
# Per cell, exactly the three stages a wave-1 cell runs:
#   1. backbone 0 -> 40 000, run_arm.sh's command line with the one flag
#      flipped to `--align-target student` (run_arm_student.sh derives it
#      textually, so nothing else can drift), seed 20260520
#   2. 2L quantile head, 15 000 steps, head seed 20260722
#   3. GIFT-Eval B4, all 97 configs
#
# Nothing past 40 000: TARGET_STEPS < FINAL_STEPS keeps run_arm.sh in its
# intermediate-wave branch, so no `_FINAL.pth` is written and no wave-2/3
# launcher can pick these up.
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train SLOTS_PER_GPU=2 \
#     nohup bash run_student_control_batch.sh > results/student_batch.log 2>&1 &
#
#   ARMS="arm5_nse arm6_v2" bash run_student_control_batch.sh   # a subset
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=gpu_pool.sh
source "$HERE/gpu_pool.sh"

# arm5 already ran (results/student_control_arm5.log, 1.4501). The other nine.
ARMS="${ARMS:-arm5_tr1 arm5_nse arm5_ncpc arm5_combab arm6_v2 arm6_v2_tr1 arm6_v2_nse arm6_v2_ncpc arm6_v2_combab}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
BB_STEP_K="${BB_STEP_K:-40}"
HEAD_STEPS="${HEAD_STEPS:-15000}"

RES="$WT/experiments/2026-08-01_lalign_teacher/results"
EVAL_ROOT="$WT/experiments/2026-08-01_lalign_teacher/eval_gm_mase"
mkdir -p "$RES"
LOG="$RES/student_batch.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [student-batch] $*" | tee -a "$LOG"; }

ctl_job() {  # <arm> <gpu>
  local arm="$1" gpu="$2"
  local cell="${arm}_alignstudent_bb${BB_STEP_K}k_hd${HEAD_STEPS}s"
  if [ -f "$EVAL_ROOT/${cell}_summary.txt" ]; then
    say "SKIP $arm — $cell already has a 97-config summary"
    return 0
  fi
  say "START $arm on GPU $gpu"
  # Each cell gets its own generated launcher. run_arm_student.sh writes the
  # sed output to $GEN and then execs it; four cells sharing one path would
  # race, and the loser would exec a half-written file.
  GEN="$HERE/.run_arm_student.${arm}.generated.sh" \
  WT="$WT" ARM="$arm" BB_GPU="$gpu" \
  BB_STEP_K="$BB_STEP_K" HEAD_STEPS="$HEAD_STEPS" \
    bash "$HERE/run_student_control.sh"
  local rc=$?
  say "END   $arm rc=$rc"
  return $rc
}

read -r -a ARM_LIST <<< "$ARMS"
say "${#ARM_LIST[@]} cells, ${SLOTS_PER_GPU} slot(s) per GPU, bb=${BB_STEP_K}k head=${HEAD_STEPS}: ${ARM_LIST[*]}"
POOL_RC_DIR="$RES/.pool_student_batch" pool_run "$SLOTS_PER_GPU" ctl_job "${ARM_LIST[@]}"

fail=0
for arm in "${ARM_LIST[@]}"; do
  rc="${POOL_RC[$arm]:-99}"
  agg="$(head -1 "$EVAL_ROOT/${arm}_alignstudent_bb${BB_STEP_K}k_hd${HEAD_STEPS}s_summary.txt" 2>/dev/null || echo '<no summary>')"
  say "RESULT $arm rc=$rc — $agg"
  [ "$rc" -eq 0 ] || fail=$(( fail + 1 ))
done
say "DONE — ${#ARM_LIST[@]} cells, $fail failed"
[ "$fail" -eq 0 ]
