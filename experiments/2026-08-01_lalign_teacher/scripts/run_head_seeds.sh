#!/bin/bash
# #390 review item 4 — head-seed uncertainty.
#
# Every number in this report is one head seed (20260722) on one backbone.
# The bootstrap in `eval_bootstrap.py` covers eval sampling only; it cannot
# see the head. This script retrains the q-head on the SAME frozen backbone
# under extra seeds and re-runs the full 97-config GIFT-Eval, so the report
# can state the spread a single cell carries. Every gap the report claims
# has to clear that spread.
#
# Nothing but `--seed` changes: same backbone checkpoint, same head steps,
# same eval. `eval_arm.sh` takes the seed from HEAD_SEED and the output
# directory from CELL_TAG, both defaulted to the waves' own values.
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train SEEDS="20260723 20260724" \
#     SLOTS_PER_GPU=1 bash run_head_seeds.sh
#
#   CELLS="arm5_nse@200@30000"  bash run_head_seeds.sh   # a subset
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=gpu_pool.sh
source "$HERE/gpu_pool.sh"

# The four cells the review named: the three that reached wave 3, plus the
# arm6_v2 base cell whose 40k → 100k jump the review flagged. All four are
# 30 000-step heads, so their spreads are directly comparable.
CELLS="${CELLS:-arm5_nse@200@30000 arm6_v2_combab@200@30000 arm6_v2_ncpc@200@30000 arm6_v2@100@30000}"
SEEDS="${SEEDS:-20260723 20260724}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"

RES="$WT/experiments/2026-08-01_lalign_teacher/results"
mkdir -p "$RES"
LOG="$RES/head_seeds.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [head-seeds] $*" | tee -a "$LOG"; }

seed_job() {  # <arm>@<bb_k>@<head_steps>@<seed>  <gpu>
  local spec="$1" gpu="$2"
  local arm bbk hds seed
  IFS='@' read -r arm bbk hds seed <<< "$spec"
  local tag="_s${seed}"
  say "START $arm bb${bbk}k hd${hds} seed=$seed on GPU $gpu"
  CF390_NAME_SUFFIX=alignteacher CELL_TAG="$tag" HEAD_SEED="$seed" \
    WT="$WT" ARM="$arm" BB_GPU="$gpu" \
    BB_STEP_K="$bbk" HEAD_STEPS="$hds" \
    bash "$HERE/eval_arm.sh"
  local rc=$?
  say "END   $arm bb${bbk}k seed=$seed rc=$rc"
  return $rc
}

JOBS=()
for cell in $CELLS; do
  for seed in $SEEDS; do JOBS+=("${cell}@${seed}"); done
done

say "${#JOBS[@]} jobs, ${SLOTS_PER_GPU} slot(s) per GPU: ${JOBS[*]}"
POOL_RC_DIR="$RES/.pool_head_seeds" pool_run "$SLOTS_PER_GPU" seed_job "${JOBS[@]}"

fail=0
for spec in "${JOBS[@]}"; do
  rc="${POOL_RC[$spec]:-99}"
  say "RESULT $spec rc=$rc"
  [ "$rc" -eq 0 ] || fail=$(( fail + 1 ))
done
say "DONE — ${#JOBS[@]} jobs, $fail failed"
[ "$fail" -eq 0 ]
