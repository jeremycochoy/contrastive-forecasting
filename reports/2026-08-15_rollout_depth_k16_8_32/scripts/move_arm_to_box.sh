#!/bin/bash
# #401 — move ONE arm from elisa to the box without losing a step.
#
# Two arms started on elisa GPU 0 while the box was rented. They hold real
# steps, and those steps stay. This script copies one arm's resume bundle to
# the box, so the box's leg resumes the arm instead of starting it again.
#
# ---- Why this script, and not scripts/push_resume_bundle.sh -------------------
#
# CLAUDE.md names `scripts/push_resume_bundle.sh` as the resume-bundle path.
# That script's RULE applies here and this script obeys it: push EVERY file
# class in one go, atomically, behind a preflight gate that refuses a
# bundle with a missing class. Its PATHS do not apply here. It reads
# `<arm dir>/checkpoints/` and `<arm dir>/run.log` and it writes
# `/workspace/app/checkpoints`, all three hard-coded — the #10-era vast.ai
# layout. This study lays a leg out as `<root>/k<K>/<cell>/leg_<N>k/` and
# names its files `<run>_<N>k.pth`, and it trains no head on the box, which
# `push_resume_bundle.sh` requires. So the rule is reused and the paths are
# this study's.
#
# ---- The four classes, and why each one is required --------------------------
#
#   <run>_<N>k.pth            the weights
#   <run>_<N>k_optimizer.pth  step counter, AdamW momentum, RNG state.
#                             `load_training_state` FALLS BACK when this file
#                             is absent: the run restarts at step 0 with a
#                             fresh optimizer and says so in one log line.
#                             That is the silent loss this gate exists for.
#   <run>_losses.csv          the curve up to the move. Nothing else holds it.
#   run_<run>.log             the trainer's own command lines, which
#                             `cf401_leg_reduce` reads to prove the objective.
#
# `_best_gap`, `_best_loss`, and the two diagnostic CSVs ride along. They are
# not required: a missing best-so-far costs a comparison, not a step.
#
# ---- Where the bundle lands, and why not in the leg it came from --------------
#
# The bundle goes to `<box root>/k<K>/<cell>/leg_<N>k/`, named for the step it
# carries, NOT into `leg_40k/` where the 40,000-step leg saves. train.py
# branches `--run-name` to `<name>_r2` when its save dir already holds
# `<name>_*.pth` (`safe_run_name`), so a bundle dropped in the target leg dir
# would produce `<name>_r2_40k.pth` and a second losses CSV. `newest_ckpt`
# globs `<cell>/leg_*/`, so the resume still finds the bundle one directory
# over, and the leg writes the canonical name.
#
# This is also how the study already splits a curve: every leg writes its own
# losses CSV in its own directory.
#
# Usage:
#   bash scripts/move_arm_to_box.sh <k> <host> <port>
#   CF401_DRY_RUN=1 bash scripts/move_arm_to_box.sh 8 ssh2.vast.ai 16048
set -uo pipefail

K="${1:?usage: move_arm_to_box.sh <k> <ssh host> <ssh port>}"
HOST="${2:?usage: move_arm_to_box.sh <k> <ssh host> <ssh port>}"
PORT="${3:?usage: move_arm_to_box.sh <k> <ssh host> <ssh port>}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
cf401_require_depth "$K" || exit $?

NAME="$(cf401_run_name "$K")"
SRC_LEG="$(cf401_leg_dir "$K" 40000)"
STUDY_REL="reports/$(basename "$CF401_STUDY")"
BOX_CF="${BOX_CF:-/root/cf}"
BOX_RES="$BOX_CF/$STUDY_REL/results/$CF401_REDUCE"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20)
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [move k=$K] $*"; }

[ -d "$SRC_LEG" ] || { echo "ABORT: no leg dir at $SRC_LEG" >&2; exit 2; }

# ---- Which checkpoint moves ---------------------------------------------------
# The furthest PERIODIC save, by the step in its name. `_best_gap` and
# `_best_loss` are not periodic: they carry whichever step scored best, which
# is behind the run and unknown to the reader.
SRC_PTH="$(ls "$SRC_LEG/$NAME"_[0-9]*k.pth 2>/dev/null \
  | grep -v optimizer \
  | sed -E 's|.*_([0-9]+)k\.pth$|\1 &|' | sort -k1,1n | tail -1 | cut -d' ' -f2-)"
[ -n "$SRC_PTH" ] || {
  echo "ABORT: no periodic checkpoint under $SRC_LEG." >&2
  echo "  The arm has not reached its first save (every 20,000 steps)." >&2
  echo "  Moving it now would throw away every step it holds." >&2
  exit 3; }

STEP_K="$(printf '%s\n' "$SRC_PTH" | sed -E 's|.*_([0-9]+)k\.pth$|\1|')"
DST_LEG="$(cf401_arm_root "$K")"
DST_LEG="${DST_LEG/#$CF401_ROOT/$CF401_BOX_RUNS}/$CF401_CELL/leg_${STEP_K}k"
SRC_LOG="$CF401_RESULTS/run_$NAME.log"

# ---- The bundle ---------------------------------------------------------------
# `<label>|<local path>|<remote path>|<required>`
BUNDLE=(
  "backbone|$SRC_PTH|$DST_LEG/$(basename "$SRC_PTH")|1"
  "backbone optimizer|${SRC_PTH%.pth}_optimizer.pth|$DST_LEG/$(basename "${SRC_PTH%.pth}")_optimizer.pth|1"
  "losses CSV|$SRC_LEG/${NAME}_losses.csv|$DST_LEG/${NAME}_losses.csv|1"
  "leg log|$SRC_LOG|$BOX_RES/run_$NAME.log|1"
  "best gap|$SRC_LEG/${NAME}_best_gap.pth|$DST_LEG/${NAME}_best_gap.pth|0"
  "best gap optimizer|$SRC_LEG/${NAME}_best_gap_optimizer.pth|$DST_LEG/${NAME}_best_gap_optimizer.pth|0"
  "best loss|$SRC_LEG/${NAME}_best_loss.pth|$DST_LEG/${NAME}_best_loss.pth|0"
  "best loss optimizer|$SRC_LEG/${NAME}_best_loss_optimizer.pth|$DST_LEG/${NAME}_best_loss_optimizer.pth|0"
  "attn amplitude CSV|$SRC_LEG/${NAME}_attn_amplitude.csv|$DST_LEG/${NAME}_attn_amplitude.csv|0"
  "latent drift CSV|$SRC_LEG/${NAME}_latent_drift.csv|$DST_LEG/${NAME}_latent_drift.csv|0"
)

echo "=== move_arm_to_box.sh ==="
echo "  arm        : k=$K reduce=$CF401_REDUCE"
echo "  run name   : $NAME"
echo "  from       : $SRC_LEG (elisa)"
echo "  resume at  : ${STEP_K}k steps"
echo "  to         : root@$HOST:$PORT $DST_LEG"
echo "  log to     : $BOX_RES/run_$NAME.log"
echo

# ---- Preflight: every required class, before one byte moves -------------------
missing=0
for row in "${BUNDLE[@]}"; do
  IFS='|' read -r label lp rp req <<<"$row"
  if [ -f "$lp" ]; then
    printf '  ok  %-22s %10d B  %s\n' "$label" "$(stat -c%s "$lp")" "$lp"
  elif [ "$req" = "1" ]; then
    printf '  ✗   %-22s MISSING     %s\n' "$label" "$lp"; missing=$(( missing + 1 ))
  else
    printf '  --  %-22s absent      %s\n' "$label" "$lp"
  fi
done
[ "$missing" -eq 0 ] || {
  echo >&2
  echo "ABORT: $missing required file(s) missing. Nothing moved." >&2
  exit 4; }

[ -n "${CF401_DRY_RUN:-}" ] && { echo; echo "(dry run — nothing pushed)"; exit 0; }

# ---- Push, atomically ---------------------------------------------------------
# Every file lands as `<remote>.tmp` and is moved into place. A dropped
# transfer therefore cannot be read as a checkpoint by a trainer that starts
# while this runs.
echo
rsh "mkdir -p '$DST_LEG' '$BOX_RES'" || {
  echo "ABORT: cannot make $DST_LEG on the box" >&2; exit 5; }

pushed=0
declare -A SENT_BYTES=()
for row in "${BUNDLE[@]}"; do
  IFS='|' read -r label lp rp req <<<"$row"
  [ -f "$lp" ] || continue
  say "push $label"
  # The size at push time, not at verify time. The losses CSV and the leg log
  # are append-only and the trainer still writes them while this runs, so a
  # local stat taken afterwards always exceeds what scp sent.
  SENT_BYTES["$label"]="$(stat -c%s "$lp")"
  scp "${SSH_OPTS[@]}" -P "$PORT" -q "$lp" "root@$HOST:$rp.tmp" || {
    echo "ABORT: scp failed for $label ($lp)" >&2; exit 5; }
  rsh "mv -f '$rp.tmp' '$rp'" || {
    echo "ABORT: remote mv failed for $label" >&2; exit 5; }
  pushed=$(( pushed + 1 ))
done

# ---- Verify by size, on the box -----------------------------------------------
echo
say "verifying $pushed file(s) on the box"
bad=0
for row in "${BUNDLE[@]}"; do
  IFS='|' read -r label lp rp req <<<"$row"
  [ -f "$lp" ] || continue
  want="${SENT_BYTES[$label]:-$(stat -c%s "$lp")}"
  got="$(rsh "stat -c%s '$rp' 2>/dev/null" | tr -dc '0-9')"
  # The box must hold at least what scp sent. An append-only file that grew
  # mid-transfer lands larger; a dropped transfer lands smaller.
  if [ -n "$got" ] && [ "$got" -ge "$want" ] 2>/dev/null; then
    printf '  ok  %-22s %10d B sent, box has %d\n' "$label" "$want" "$got"
  else
    printf '  ✗   %-22s sent %d B, box has %s\n' "$label" "$want" "${got:-nothing}"
    bad=$(( bad + 1 ))
  fi
done
[ "$bad" -eq 0 ] || { echo "ABORT: $bad file(s) did not land." >&2; exit 6; }

echo
say "OK — k=$K resumes on the box from step ${STEP_K}000"
echo "  the box's leg finds it through newest_ckpt over $DST_LEG"
