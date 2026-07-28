#!/bin/bash
# #382 — 100-step smoke on all 8 arms, one arm at a time on GPU 0.
#
# Catches config-typo bugs before the full 100k-step vast run. Writes
# under a temporary results/artifacts subtree ("smoke") so it never
# collides with production artifacts.
set -uo pipefail

WT="${WT:?WT (worktree root) must be set}"
GPU="${GPU:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$WT/experiments/2026-07-28_loss_term_isolation}"

ARMS=(pred rep align pred_moco rep_moco sigreg_e sigreg_h cpc)
export OUT="$OUT/smoke"
mkdir -p "$OUT"/{artifacts,results}
for a in "${ARMS[@]}"; do mkdir -p "$OUT/artifacts/$a"; done

failed=()
for arm in "${ARMS[@]}"; do
  echo "[smoke] arm=$arm"
  OUT="$OUT" bash "$HERE/run_arm.sh" "$arm" "$GPU" 100 100 || failed+=("$arm")
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[smoke] FAILED arms: ${failed[*]}"
  exit 1
fi
echo "[smoke] all 8 arms passed 100-step smoke"
