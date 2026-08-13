#!/bin/bash
# #373 review gap 3 — what the prepared control row would do, without doing it.
#
# gap3_jobs.tsv holds one row and nothing runs it. This prints the run name,
# the directory, the checkpoint the worker would wait for, and the exact
# trainer flags, and it checks the target does not collide with a checkpoint
# already on disk. Read this before spending the GPU-hours.
#
# Usage: bash gap3_preflight.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
JOBS="${JOBS:-$HERE/gap3_jobs.tsv}"

while IFS=$'\t' read -r id gap launcher arg k suf seed gargs steps; do
  case "$id" in ''|\#*) continue ;; esac
  [ "$suf"   = "-" ] && suf=""
  [ "$gargs" = "-" ] && gargs=""

  case "$launcher" in
    run_leg_k.sh) name="cf393_${arg}_cf373k${k}${suf}"
                  dir="$CF373_ROOT/$arg/leg_$(( steps / 1000 ))k" ;;
    *)            name="$(awk -v a="  $arg)" '
                    $0 == a { grab = 1; next }
                    grab && /^ *NAME=/ { sub(/^ *NAME="/, ""); sub(/"$/, "");
                                         print; exit }' "$HERE/$launcher")"
                  [ -n "$name" ] || { echo "ABORT: no NAME for arm '$arg'"; exit 2; }
                  name="${name}_cf373k${k}${suf}"
                  dir="$CF373_ROOT/$name" ;;
  esac
  ck="$dir/${name}_$(( steps / 1000 ))k.pth"

  echo "id         $id  (review gap $gap)"
  echo "launcher   $launcher $arg"
  echo "flags      K=$k SEED=$seed RUN_SUFFIX='$suf' GAP_ARGS='$gargs' TARGET_STEPS=$steps"
  echo "run name   $name"
  echo "directory  $dir"
  echo "checkpoint $ck"
  if [ -e "$dir" ]; then
    echo "COLLISION  $dir already exists — pick another suffix"
  else
    echo "collision  none: $dir is free"
  fi
  echo "heads      ${id}_bb40k_student and ${id}_bb40k_teacher, 15,000 steps, seed 20260722"
  echo "scores     results/score_${id}_bb40k_student.txt, results/score_${id}_bb40k_teacher.txt"
  echo "launch     JOBS=scripts/gap3_jobs.tsv BB_GPU=<0|1> \\"
  echo "             nohup bash scripts/gap_worker.sh 0 1 > results/gap3_worker.out 2>&1 &"
  echo
done < "$JOBS"

echo "Nothing was started. This script trains nothing and rents nothing."
