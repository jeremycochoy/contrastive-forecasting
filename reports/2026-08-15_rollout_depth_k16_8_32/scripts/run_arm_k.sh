#!/bin/bash
# #401 — one leg of one arm: train the cell to <stop> steps at depth <k>.
#
# This is a wrapper, on purpose. The trainer command line for this
# configuration lives in ONE place, #373's `run_leg_k.sh`, and a copy of it
# here would be a second protocol that drifts. The wrapper supplies four
# things #373's runner takes from the environment:
#
#   K              the rollout depth, 16 / 8 / 32
#   RUNS           this study's durable root, so no checkpoint mixes with a
#                  #373 one
#   CF_STUDY_DIR   this study's directory, so the leg's log lands here
#   BB_GPU         the card
#
# The runner is idempotent per leg: a stop whose checkpoint is on disk is a
# no-op, and a leg resumes the cell's furthest checkpoint with its optimizer
# state. So a re-fired stop after a crash costs nothing.
#
# Usage:  run_arm_k.sh <k> <stop steps>
#         BB_GPU=0 bash run_arm_k.sh 16 40000
#         CF401_DRY_RUN=1 bash run_arm_k.sh 16 40000     # print, do not run
set -uo pipefail

K="${1:?usage: run_arm_k.sh <k> <stop steps>}"
STOP="${2:?usage: run_arm_k.sh <k> <stop steps>}"

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"
cf401_require_depth "$K" || exit $?
cf401_require_stop "$STOP" || exit $?

RUNNER="$CF401_PARENT/scripts/run_leg_k.sh"
[ -f "$RUNNER" ] || { echo "ABORT: no runner at $RUNNER" >&2; exit 2; }

BB_GPU="${BB_GPU:-0}"
mkdir -p "$CF401_RESULTS"

ARM_ROOT="$(cf401_arm_root "$K")"

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "arm cell=$CF401_CELL K=$K steps=$STOP gpu=$BB_GPU"
  echo "  runner=$RUNNER"
  echo "  RUNS=$ARM_ROOT CF_STUDY_DIR=$CF401_STUDY"
  echo "  ckpt=$(cf401_leg_dir "$K" "$STOP")/$(cf401_run_name "$K")_$(( STOP / 1000 ))k.pth"
  exit 0
fi

echo "[$(date '+%m-%d %H:%M:%S')] [#401] arm k=$K -> ${STOP} steps on gpu $BB_GPU" \
  | tee -a "$CF401_RESULTS/arms.log"
K="$K" RUNS="$ARM_ROOT" CF_STUDY_DIR="$CF401_STUDY" WT="$CF401_WT" \
  BB_GPU="$BB_GPU" \
  bash "$RUNNER" "$CF401_CELL" "$STOP"
rc=$?
echo "[$(date '+%m-%d %H:%M:%S')] [#401] arm k=$K stop=$STOP rc=$rc" \
  | tee -a "$CF401_RESULTS/arms.log"
exit $rc
