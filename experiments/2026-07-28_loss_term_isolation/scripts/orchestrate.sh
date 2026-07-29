#!/bin/bash
# #382 — pipeline the 8 arms across the vast instance's GPUs.
#
# Runs $NGPUS arms in parallel (one arm per GPU under Exclusive_Process)
# and re-launches the next arm on each GPU as the previous one finishes.
# Blocks until all 8 arms complete.
#
# Env:
#   WT         — worktree root (required; forwarded to run_arm.sh).
#   NGPUS      — how many GPUs are available. Default: auto-detect.
#   STEPS      — --total-steps forwarded to run_arm.sh. Default 100000.
#   SAVE_EVERY — --save-every forwarded to run_arm.sh. Default 5000.
set -uo pipefail

WT="${WT:?WT (worktree root) must be set}"
STEPS="${STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "$NGPUS" -ge 1 ] || { echo "orchestrate.sh: no GPUs available (NGPUS=$NGPUS)"; exit 1; }

ARMS=(pred rep align pred_moco rep_moco sigreg_e sigreg_h cpc)
HERE="$(cd "$(dirname "$0")" && pwd)"

# Per-GPU worker: sequential loop over the arms allocated to that GPU.
gpu_worker() {
  local gpu="$1"; shift
  for arm in "$@"; do
    echo "[gpu=$gpu] launching arm=$arm ($(date +%m-%d-%H:%M))"
    bash "$HERE/run_arm.sh" "$arm" "$gpu" "$STEPS" "$SAVE_EVERY"
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "[gpu=$gpu] arm=$arm FAILED rc=$rc — moving on"
    fi
  done
}

# Round-robin arms across GPUs so each GPU gets a balanced share.
declare -a gpu_arms
for i in "${!ARMS[@]}"; do
  gpu=$(( i % NGPUS ))
  gpu_arms[$gpu]="${gpu_arms[$gpu]:-} ${ARMS[$i]}"
done

for gpu in $(seq 0 $((NGPUS - 1))); do
  echo "gpu=$gpu ← arms:${gpu_arms[$gpu]}"
  # shellcheck disable=SC2086
  gpu_worker "$gpu" ${gpu_arms[$gpu]} &
done
wait
echo "orchestrate.sh: all workers exited ($(date +%m-%d-%H:%M))"
