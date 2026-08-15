#!/bin/bash
# #401 — does #373's sync loop reach this study's checkpoints?
#
# #373 saves at `$REMOTE_RUNS/<cell>/leg_<N>k/`. This study saves one level
# deeper, at `$REMOTE_RUNS/k<K>/<cell>/leg_<N>k/`, because `cf401_arm_root`
# adds the depth (one root per arm, CLAUDE.md checkpoint safety rule 3). A
# loop that globbed a fixed depth below `REMOTE_RUNS` would pull nothing, and
# `launch_sync.sh`'s first-tick check would report it only after 10 minutes.
#
# So the loop's own listing command is taken OUT of `sync_loop.sh` — not
# copied here, read from there, so a change to it is caught — and run against
# a tree with this study's layout. Every artefact class must come back.
#
# Usage:  bash sync/verify_glob_depth.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$(dirname "$HERE")/scripts/study.sh"

LOOP="$CF401_PARENT/sync/sync_loop.sh"
[ -f "$LOOP" ] || { echo "ABORT: no sync loop at $LOOP" >&2; exit 2; }

# The one line of `remote_listing` that walks the tree. A `-maxdepth` in it
# is the failure this script exists to find.
FIND_LINE="$(grep -n "find '\$1' -type f" "$LOOP" | head -1)"
[ -n "$FIND_LINE" ] || {
  echo "FAIL: sync_loop.sh no longer lists with \`find '\$1' -type f\`" >&2
  echo "  Re-read remote_listing() in $LOOP before trusting this study's sync." >&2
  exit 1; }
case "$FIND_LINE" in
  *-maxdepth*) echo "FAIL: the loop bounds its walk: $FIND_LINE" >&2; exit 1 ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
RUNS="$TMP/runs"

# This study's layout, one file per class the sync rules require.
for k in $CF401_DEPTHS; do
  for stop in $CF401_STOPS; do
    leg="$RUNS/k$k/$CF401_CELL/leg_$(( stop / 1000 ))k"
    mkdir -p "$leg"
    name="cf393_${CF401_CELL}_cf373k${k}_$(( stop / 1000 ))k"
    : >"$leg/$name.pth"
    : >"$leg/${name}_optimizer.pth"
    : >"$leg/cf393_${CF401_CELL}_cf373k${k}_losses.csv"
  done
  tag="k${k}_bb40k_h30k_$CF401_ENC"
  mkdir -p "$RUNS/k$k/eval/$tag/gift"
  : >"$RUNS/k$k/eval/$tag/qhead_${tag}_s20260722_final.pth"
  : >"$RUNS/k$k/eval/$tag/gift/all_results.csv"
done
want=$(find "$RUNS" -type f | wc -l)

# The loop's own command, with $1 bound to the root, run locally.
got="$(bash -c "find '$RUNS' -type f -printf '%s %p\n' 2>/dev/null" | wc -l)"

echo "layout: $RUNS"
echo "files on the tree: $want    found by the loop's listing: $got"
[ "$got" -eq "$want" ] || {
  echo "FAIL: the loop's listing missed $(( want - got )) file(s)" >&2; exit 1; }

# The local path a pulled file takes, the same way pull_tree computes it.
deep="$(find "$RUNS" -name '*_optimizer.pth' | head -1)"
rel="${deep#$RUNS/}"
case "$rel" in
  k*/"$CF401_CELL"/leg_*k/*) ;;
  *) echo "FAIL: a pulled file would land at '$rel', not under k<K>/" >&2
     exit 1 ;;
esac
echo "deepest artefact lands at sync/$rel"
echo "PASS: REMOTE_RUNS=\$CF401_ROOT reaches every arm at every depth"
