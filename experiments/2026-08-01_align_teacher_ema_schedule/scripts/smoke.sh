#!/bin/bash
# #388 pre-launch smoke: 20 steps per arm on GPU, into a throwaway dir.
#
# Catches flag typos, argparse rejections and shape errors before the four
# ~4h runs start. Asserts the two things #388 adds actually land in the
# CSVs: an `ema_tau` column that moves under the schedule, and drift rows
# for both `student_h` and `teacher_h`.
#
# Usage: WT=/tmp/contrastive-forecasting-388 smoke.sh <gpu>
set -uo pipefail

GPU="${1:-0}"
WT="${WT:?WT (worktree root) must be set}"
TMP="${TMP_SMOKE:-/tmp/cf388_smoke}"
rm -rf "$TMP"; mkdir -p "$TMP"

fail=0
for arm in align_teacher_a09 align_teacher_sched pred_moco_sched rep_moco_sched; do
  RUNS="$TMP/runs" OUT="$TMP/out" WT="$WT" PROBE_EVERY=5 \
    bash "$WT/experiments/2026-08-01_align_teacher_ema_schedule/scripts/run_arm.sh" \
    "$arm" "$GPU" 20 20
  rc=$?
  name="ats_${arm}"
  losses="$TMP/runs/$arm/${name}_losses.csv"
  drift="$TMP/runs/$arm/${name}_latent_drift.csv"
  if [ $rc -ne 0 ]; then echo "SMOKE FAIL $arm: rc=$rc"; fail=1; continue; fi
  python3 - "$arm" "$losses" "$drift" <<'PY'
import csv, sys
arm, losses, drift = sys.argv[1:4]
rows = list(csv.DictReader(open(losses)))
alphas = sorted({float(r["ema_tau"]) for r in rows})
assert alphas, f"{arm}: no ema_tau values"
if arm.endswith("_sched"):
    assert len(alphas) > 1, f"{arm}: scheduled alpha did not move: {alphas}"
else:
    assert alphas == [0.9], f"{arm}: constant alpha expected 0.9, got {alphas}"
lat = {r["latent"] for r in csv.DictReader(open(drift))}
assert lat == {"student_h", "teacher_h"}, f"{arm}: drift latents {lat}"
print(f"SMOKE OK {arm}: alpha {alphas[0]:.4f}..{alphas[-1]:.4f}, latents {sorted(lat)}")
PY
  [ $? -ne 0 ] && fail=1
done

[ $fail -eq 0 ] && { echo "SMOKE PASS (all 4 arms)"; exit 0; }
echo "SMOKE FAILED"; exit 1
