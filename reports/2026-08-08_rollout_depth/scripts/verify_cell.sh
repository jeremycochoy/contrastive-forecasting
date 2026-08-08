#!/bin/bash
# #373 — prove one cell's launcher hands the depth to the trainer.
#
# Runs the cell's OWN launcher twice for one step, at k = 0 and at k = 3,
# then hands the two first CSV rows to check_depth_reached.py. Running the
# launcher rather than a hand-written command line is the point: the thing
# under test is the launcher, and a per-cell `case` arm that reassigns
# EXTRA_ARGS is exactly the failure this catches.
#
# One step, because that is the discriminating row. Both runs start from the
# same seed and draw the same first batch, so `loss_tau_ref` (pinned to
# depth 0) must match and `loss` must not. By step 200 the weights have
# diverged and neither column proves anything.
#
# Usage: bash verify_cell.sh <cell id>          # A1..A4, B1..B10
#        BB_GPU=1 bash verify_cell.sh B5
set -uo pipefail

CELL_ID="${1:?usage: verify_cell.sh <cell id>}"
K="${K:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${WT:-/home/jupyter/wt-cf-373-train}"
OUT="$WT/reports/2026-08-08_rollout_depth"
RES="$OUT/results"
SCRATCH="${CF373_VERIFY_RUNS:-/home/jupyter/checkpoints_backup/cf-373/verify}"
mkdir -p "$RES" "$SCRATCH"

row="$(awk -F'\t' -v c="$CELL_ID" '$1==c {print; exit}' "$HERE/cells.tsv")"
[ -n "$row" ] || { echo "ABORT: no cell '$CELL_ID' in cells.tsv" >&2; exit 2; }
SLUG=$(cut -f2 <<<"$row"); LAUNCHER=$(cut -f3 <<<"$row"); ARG=$(cut -f4 <<<"$row")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [verify $CELL_ID] $*" \
  | tee -a "$RES/verify_${CELL_ID}.log"; }

# One depth per invocation, into its own scratch root, so neither run can
# resume the other's checkpoint or read its CSV.
run_one(){ # <k>
  local k="$1"
  local root="$SCRATCH/${CELL_ID}_k${k}"
  rm -rf "$root"; mkdir -p "$root"
  local args=("$ARG")
  case "$LAUNCHER" in
    run_leg_k.sh) args+=(1) ;;   # <cell> <target steps>
  esac
  K="$k" LOG_EVERY=1 TARGET_STEPS=1 FINAL_STEPS=200000 \
  EXTRA_SAVES=1 SAVE_EVERY=1000000 \
  WT="$WT" BB_GPU="${BB_GPU:-1}" RUNS="$root" CF373_RUNS="$root" \
    bash "$WT/reports/2026-08-08_rollout_depth/scripts/$LAUNCHER" "${args[@]}" \
    >>"$RES/verify_${CELL_ID}.log" 2>&1
  local rc=$?
  # The trainer writes <run name>_losses.csv beside its checkpoints. The
  # run name is the launcher's, so find it rather than rebuild it here.
  local csv; csv="$(find "$root" -name '*_losses.csv' | head -1)"
  printf '%s\t%s\n' "$rc" "$csv"
}

log "START slug=$SLUG launcher=$LAUNCHER arg=$ARG gpu=${BB_GPU:-1}"

# The launcher's exit code is not the gate here. A one-step invocation makes
# both launchers look for a `_0k.pth` snapshot that a 1-step run never names,
# so they report a missing checkpoint and exit non-zero after a training step
# that ran fine. The gate is the CSV: no row means no step.
IFS=$'\t' read -r rc0 csv0 < <(run_one 0)
[ -s "${csv0:-/nonexistent}" ] \
  || { log "FAIL: k=0 run wrote no losses CSV (launcher rc=$rc0)"; exit 1; }
IFS=$'\t' read -r rc3 csv3 < <(run_one "$K")
[ -s "${csv3:-/nonexistent}" ] \
  || { log "FAIL: k=$K run wrote no losses CSV (launcher rc=$rc3)"; exit 1; }
log "launcher rc: k=0 -> $rc0, k=$K -> $rc3 (a 1-step run names no _0k.pth)"

# Keep the two CSVs: they are the evidence, and each is a few hundred bytes.
cp -f "$csv0" "$RES/verify_${CELL_ID}_k0_losses.csv"
cp -f "$csv3" "$RES/verify_${CELL_ID}_k${K}_losses.csv"

python3 "$HERE/check_depth_reached.py" "$CELL_ID" "$csv0" "$csv3" "$K" \
  | tee -a "$RES/verify_${CELL_ID}.log" "$RES/verify_summary.tsv" >/dev/null
rc=${PIPESTATUS[0]}
tail -1 "$RES/verify_${CELL_ID}.log"
log "verdict rc=$rc"
exit "$rc"
