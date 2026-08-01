#!/bin/bash
# #388 — launch the four runs, two per GPU, and wait.
#
# Usage: WT=/tmp/contrastive-forecasting-388 orchestrate.sh [total_steps]
#
# elisa has two RTX 4090s. Each arm is the small #379 backbone (720k
# params, ~5 MB checkpoint), so two arms share a card comfortably.
set -uo pipefail

STEPS="${1:-100000}"
WT="${WT:?WT (worktree root) must be set}"
OUT="$WT/experiments/2026-08-01_align_teacher_ema_schedule"
RES="$OUT/results"
mkdir -p "$RES"

declare -A GPU_OF=(
  [align_teacher_a09]=0
  [align_teacher_sched]=0
  [pred_moco_sched]=1
  [rep_moco_sched]=1
)

pids=()
for arm in align_teacher_a09 align_teacher_sched pred_moco_sched rep_moco_sched; do
  WT="$WT" bash "$OUT/scripts/run_arm.sh" "$arm" "${GPU_OF[$arm]}" "$STEPS" 5000 \
      >>"$RES/orchestrate.log" 2>&1 &
  pids+=($!)
  echo "[$(date +%m-%d-%H:%M)] launched $arm on gpu ${GPU_OF[$arm]} (pid ${pids[-1]})" \
      | tee -a "$RES/orchestrate.log"
  sleep 20   # stagger the HF stream handshakes
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[$(date +%m-%d-%H:%M)] all arms finished, rc=$rc" | tee -a "$RES/orchestrate.log"
exit $rc
