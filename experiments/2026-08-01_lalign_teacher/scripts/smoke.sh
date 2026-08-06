#!/bin/bash
# #390 backbone-only smoke — one arm's backbone for SMOKE_STEPS steps, so
# any breakage in the training config, checkpoint naming, or
# training-dynamics logging surfaces BEFORE the 40k/100k/200k waves are
# committed. No q-head, no eval — matches the real experiment shape.
#
# Exercises:
#   1. The arm's config loads and runs — including `--align-target teacher`
#      reaching the add-on inside contrastive_latent_loss on REAL data.
#   2. `--extra-save-steps` snapshot lands next to the save-every snapshot.
#   3. `_losses.csv` carries the `ff` / `u_batchtime` / `u_batchtime_e`
#      columns the plot scripts read.
#   4. Step time — printed as `sps`, so the per-step cost of `arm6_v2`
#      (teacher latent in the MoCo keys AND in L_align) is measured against
#      `arm5` rather than assumed.
#
# Any failure short-circuits with a non-zero exit and a clear message.
# Success prints `SMOKE OK` and a one-line summary including sps.
#
# Usage (from the checkout root):
#   ARM=arm6_v2 GPU=1 bash experiments/2026-08-01_lalign_teacher/scripts/smoke.sh
#
# Any of the 10 arms in run_arm.sh is valid. Run `arm6_v2` first: it is the
# heavier of the two families (`--moco-rep-keys` on top of the align term).
set -euo pipefail

ARM="${ARM:-arm6_v2}"
GPU="${GPU:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-200}"
EXPDIR="2026-08-01_lalign_teacher"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

# WT for the smoke — under $HOME so it passes run_arm.sh's under-/tmp guard
# and, in the same stroke, is not the throwaway agent worktree that real
# launches ban (CLAUDE.md § Checkpoint Safety Rule 4).
SMOKE_WT="${HOME}/.cache/cf-390-smoke-$$"
cleanup(){ rm -rf "$SMOKE_WT"; }
trap cleanup EXIT
mkdir -p "$SMOKE_WT"

# Layer the smoke WT over the real repo: symlink every top-level dir so the
# launcher finds train.py and the HF token, but override this experiment's
# dir with a fresh one so runs/ and results/ are empty and writable without
# polluting the real checkout.
for d in src scripts docs tests reports; do
  [ -e "$REPO_ROOT/$d" ] && ln -s "$REPO_ROOT/$d" "$SMOKE_WT/$d"
done
mkdir -p "$SMOKE_WT/experiments"
for exp in "$REPO_ROOT/experiments"/*; do
  base="$(basename "$exp")"
  [ "$base" = "$EXPDIR" ] || ln -s "$exp" "$SMOKE_WT/experiments/$base"
done
mkdir -p "$SMOKE_WT/experiments/$EXPDIR"
ln -s "$REPO_ROOT/experiments/$EXPDIR/scripts" "$SMOKE_WT/experiments/$EXPDIR/scripts"
[ -e "$REPO_ROOT/experiments/$EXPDIR/sync" ] && \
  ln -s "$REPO_ROOT/experiments/$EXPDIR/sync" "$SMOKE_WT/experiments/$EXPDIR/sync"
[ -e "$REPO_ROOT/experiments/$EXPDIR/README.md" ] && \
  ln -s "$REPO_ROOT/experiments/$EXPDIR/README.md" "$SMOKE_WT/experiments/$EXPDIR/README.md"

TOK="$REPO_ROOT/experiments/hf_token.txt"
[ -f "$TOK" ] || {
  echo "[smoke-390] ABORT: HF token missing at $TOK — an anonymous HF stream" \
       "is rate-limited and idles the GPU (CLAUDE.md § HuggingFace token)." >&2
  exit 2
}

echo "[smoke-390] SMOKE_WT=$SMOKE_WT ARM=$ARM GPU=$GPU STEPS=$SMOKE_STEPS"

# save-every = SMOKE_STEPS/2 (a regular snapshot mid-run and at the end);
# extras = 3/4 of the run (exercises the union rule + the sub-1000 `_0k.pth`
# filename). TARGET_STEPS = FINAL_STEPS so run_arm.sh treats this as a final
# wave and copies `_FINAL.pth`, the sentinel checked below.
HALF=$(( SMOKE_STEPS / 2 )); THREE_Q=$(( SMOKE_STEPS * 3 / 4 ))
TARGET_STEPS="$SMOKE_STEPS" FINAL_STEPS="$SMOKE_STEPS" \
SAVE_EVERY="$HALF" EXTRA_SAVES="$THREE_Q" \
WT="$SMOKE_WT" BB_GPU="$GPU" \
  bash "$SMOKE_WT/experiments/$EXPDIR/scripts/run_arm.sh" "$ARM" \
  > "$SMOKE_WT/smoke.log" 2>&1 || {
    echo "[smoke-390] SMOKE FAILED — tail:" >&2
    tail -30 "$SMOKE_WT/smoke.log" >&2
    exit 1
  }

RUNS="$SMOKE_WT/experiments/$EXPDIR/runs"
RES="$SMOKE_WT/experiments/$EXPDIR/results"

FINAL=$(ls "$RUNS"/*_FINAL.pth 2>/dev/null | head -1) || true
[ -n "$FINAL" ] || {
  echo "[smoke-390] SMOKE FAILED — no backbone FINAL.pth in $RUNS." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
EXTRA=$(ls "$RUNS"/*_0k.pth 2>/dev/null | head -1) || true
[ -n "$EXTRA" ] || {
  echo "[smoke-390] SMOKE FAILED — extra-save snapshot _0k.pth missing." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}
CSV=$(ls "$RUNS"/*_losses.csv 2>/dev/null | head -1) || true
[ -n "$CSV" ] || {
  echo "[smoke-390] SMOKE FAILED — losses.csv missing (training-dynamics log)." >&2
  tail -20 "$SMOKE_WT/smoke.log" >&2
  exit 1
}

# The plot scripts read ff / u_batchtime / u_batchtime_e — a missing column
# means the deliverable plots silently fail. Read the last populated row too,
# so a bare header does not pass.
#
# Take the helper's own exit status, not a wrapper's: `mapfile -t V < <(…)`
# reports mapfile's status (always 0) and `set -e` does not reach into a
# process substitution, so the exit 2 / exit 3 below would be swallowed and
# a broken CSV would print SMOKE OK.
DYN_RC=0
DYN_OUT="$(python3 - "$CSV" <<'PY'
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
)" || DYN_RC=$?
if [ "$DYN_RC" -ne 0 ]; then
  echo "[smoke-390] SMOKE FAILED — training-dynamics check on $CSV" \
       "(rc=$DYN_RC): ${DYN_OUT:-<no output>}" >&2
  exit 1
fi
mapfile -t VALS <<< "$DYN_OUT"
[ "${#VALS[@]}" -eq 3 ] || {
  echo "[smoke-390] SMOKE FAILED — training-dynamics check on $CSV returned" \
       "${#VALS[@]} values, expected 3: ${VALS[*]}" >&2
  exit 1
}

# Step time. The trainer prints `<x> sps` on every --log-every line; take the
# last one, which is past dataloader warm-up.
TLOG=$(ls "$RES"/run_*.log 2>/dev/null | head -1) || true
SPS=$(grep -oE '[0-9.]+ sps' "$TLOG" 2>/dev/null | tail -1 | cut -d' ' -f1)
[ -n "${SPS:-}" ] || SPS="?"

echo "[smoke-390] SMOKE OK — arm=$ARM"
echo "[smoke-390]   backbone: $(basename "$FINAL")"
echo "[smoke-390]   extra:    $(basename "$EXTRA")"
echo "[smoke-390]   losses:   $(basename "$CSV")  (final ff=${VALS[0]}  u_batchtime=${VALS[1]}  u_batchtime_e=${VALS[2]})"
echo "[smoke-390]   step time: $SPS sps"
