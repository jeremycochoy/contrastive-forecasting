#!/bin/bash
# #379 backbone-only smoke — runs one arm's backbone for 200 steps in
# ~3 min so any breakage in the training config, checkpoint naming, or
# training-dynamics logging surfaces BEFORE 15-20 h of GPU time is
# committed. No q-head, no eval — matches the real experiment shape.
#
# Exercises:
#   1. Training config actually loads and runs (loss / arch / SIGReg
#      combined) for the requested arm.
#   2. `--extra-save-steps 150` snapshot lands as `_0k.pth` next to the
#      regular save-every snapshot.
#   3. `_losses.csv` picks up the `ff`, `u_batchtime`, `u_batchtime_e`
#      columns the plot scripts depend on.
#
# Any failure short-circuits with a non-zero exit and a clear message.
# Success prints `SMOKE OK` and a one-line summary.
#
# Usage (from the checkout root):
#   ARM=arm1 GPU=0 bash experiments/2026-07-21_split_pred_rep_small/scripts/smoke.sh
#
# Any of {arm1, arm3, arm4, arm5, arm6_v2, bimoco} is valid. bimoco is
# the harshest recipe (--moco-negatives + --moco-rep-keys); use it once
# the arm1 smoke passes.
set -euo pipefail

ARM="${ARM:-arm1}"
GPU="${GPU:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# WT for the smoke — under $HOME so it passes run_arm.sh's under-/tmp
# guard and, in the same stroke, is not the throwaway agent worktree
# that real launches ban.
SMOKE_WT="${HOME}/.cache/cf-379-smoke-$$"
cleanup(){ rm -rf "$SMOKE_WT"; }
trap cleanup EXIT
mkdir -p "$SMOKE_WT"

# Layer the smoke WT over the real repo: symlink every top-level dir so
# the launcher finds train.py and the HF token, but override the
# experiment dir with a fresh one so runs/ and results/ are empty and
# writable without polluting the real checkout.
for d in src scripts docs tests reports pytrade; do
  [ -e "$REPO_ROOT/$d" ] && ln -s "$REPO_ROOT/$d" "$SMOKE_WT/$d"
done
mkdir -p "$SMOKE_WT/experiments"
for exp in "$REPO_ROOT/experiments"/*; do
  base="$(basename "$exp")"
  [ "$base" = "2026-07-21_split_pred_rep_small" ] || ln -s "$exp" "$SMOKE_WT/experiments/$base"
done
mkdir -p "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small"
ln -s "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/scripts" \
      "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/scripts"
ln -s "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/sync" \
      "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/sync"
[ -e "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/README.md" ] && \
  ln -s "$REPO_ROOT/experiments/2026-07-21_split_pred_rep_small/README.md" \
        "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/README.md"

TOK="$REPO_ROOT/experiments/hf_token.txt"
[ -f "$TOK" ] || {
  echo "[smoke-379] ABORT: HF token missing at $TOK — set HF_TOKEN in the env" \
       "and echo it into that file before running smoke.sh." >&2
  exit 2
}

echo "[smoke-379] SMOKE_WT=$SMOKE_WT ARM=$ARM GPU=$GPU"
echo "[smoke-379] 200-step backbone smoke — ~3 min on 4090"

# 200-step backbone. save-every=100 (regular snapshot at 100, 200);
# extras=150 (exercises the union rule + sub-1000 filename `_0k.pth`).
STEPS=200 SAVE_EVERY=100 EXTRA_SAVES="150" \
WT="$SMOKE_WT" BB_GPU="$GPU" \
  bash "$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/scripts/run_arm.sh" "$ARM" \
  > "$SMOKE_WT/smoke.log" 2>&1 || {
    echo "[smoke-379] SMOKE FAILED — tail:" >&2
    tail -30 "$SMOKE_WT/smoke.log" >&2
    exit 1
  }

# Verify the artefacts we expect are on disk.
RUNS="$SMOKE_WT/experiments/2026-07-21_split_pred_rep_small/runs"
FINAL=$(ls "$RUNS"/*_FINAL.pth 2>/dev/null | head -1) || true
[ -n "$FINAL" ] || {
  echo "[smoke-379] SMOKE FAILED — no backbone FINAL.pth in $RUNS." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
EXTRA=$(ls "$RUNS"/*_0k.pth 2>/dev/null | head -1) || true
[ -n "$EXTRA" ] || {
  echo "[smoke-379] SMOKE FAILED — extra-save snapshot _0k.pth missing." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
CSV=$(ls "$RUNS"/*_losses.csv 2>/dev/null | head -1) || true
[ -n "$CSV" ] || {
  echo "[smoke-379] SMOKE FAILED — losses.csv missing (training-dynamics log)." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
# The plot scripts read the ff / u_batchtime / u_batchtime_e columns —
# any missing column means the deliverable plots would silently fail.
# Also read the last populated row to confirm the trainer actually
# wrote numeric values (a bare header would pass the column check).
mapfile -t VALS < <(python3 - "$CSV" <<'PY'
import csv, sys
required = ("ff", "u_batchtime", "u_batchtime_e")
with open(sys.argv[1], newline="") as f:
    reader = csv.DictReader(f)
    missing = [c for c in required if c not in (reader.fieldnames or ())]
    if missing:
        print("MISSING_COLS=" + ",".join(missing))
        sys.exit(2)
    last = None
    for row in reader:
        if all(row.get(c) not in (None, "") for c in required):
            last = row
if last is None:
    print("NO_ROW")
    sys.exit(3)
for c in required:
    print(last[c])
PY
) || {
  echo "[smoke-379] SMOKE FAILED — training-dynamics check on $CSV: ${VALS[*]:-<empty>}" >&2
  exit 1
}

echo "[smoke-379] SMOKE OK — arm=$ARM"
echo "[smoke-379]   backbone: $(basename "$FINAL")"
echo "[smoke-379]   extra:    $(basename "$EXTRA")"
echo "[smoke-379]   losses:   $(basename "$CSV")  (final ff=${VALS[0]}  u_batchtime=${VALS[1]}  u_batchtime_e=${VALS[2]})"
