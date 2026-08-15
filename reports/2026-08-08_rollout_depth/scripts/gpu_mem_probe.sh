#!/bin/bash
# #373 — how much GPU memory the depth costs, on the first cell.
#
# steptime.sh reads `nvidia-smi` before and after a run, which on a shared
# card returns the neighbour's figure both times. This samples DURING the
# run instead, and reports the peak the card reached minus the floor it sat
# at just before the trainer started.
#
# Usage: bash gpu_mem_probe.sh [cell id] [steps]
set -uo pipefail

CELL_ID="${1:-B5}"
STEPS="${2:-300}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${WT:-/home/jupyter/wt-cf-373-train}"
RES="$WT/reports/2026-08-08_rollout_depth/results"
SCRATCH="${CF373_VERIFY_RUNS:-/home/jupyter/checkpoints_backup/cf-373/mem}"
GPU="${BB_GPU:-1}"
OUT="$RES/gpu_mem_${CELL_ID}.csv"
mkdir -p "$RES" "$SCRATCH"

row="$(awk -F'\t' -v c="$CELL_ID" '$1==c {print; exit}' "$HERE/cells.tsv")"
[ -n "$row" ] || { echo "ABORT: no cell '$CELL_ID'" >&2; exit 2; }
LAUNCHER=$(cut -f3 <<<"$row"); ARG=$(cut -f4 <<<"$row")

echo "cell,k,floor_mib,peak_mib,run_mib" >"$OUT"

for k in 0 3; do
  root="$SCRATCH/${CELL_ID}_k${k}"; rm -rf "$root"; mkdir -p "$root"
  args=("$ARG"); case "$LAUNCHER" in run_leg_k.sh) args+=("$STEPS") ;; esac

  floor=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  ( K="$k" LOG_EVERY=100000 TARGET_STEPS="$STEPS" FINAL_STEPS=200000 \
    EXTRA_SAVES="$STEPS" SAVE_EVERY=1000000 \
    WT="$WT" BB_GPU="$GPU" RUNS="$root" CF373_RUNS="$root" \
      bash "$WT/reports/2026-08-08_rollout_depth/scripts/$LAUNCHER" "${args[@]}" \
      >>"$RES/gpu_mem_${CELL_ID}.log" 2>&1 ) &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    m=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 2
  done
  wait "$pid"
  echo "$CELL_ID,$k,$floor,$peak,$(( peak - floor ))" | tee -a "$OUT"
  rm -rf "$root"
done

echo "--- $OUT ---"; cat "$OUT"
