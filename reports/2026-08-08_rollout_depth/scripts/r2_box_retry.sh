#!/bin/bash
# #373 round 2 — one cell, retried until a box really holds it.
#
# Usage: bash r2_box_retry.sh <cell> <stop> [stop...]
#
# `r2_launch_cell.sh` destroys a box that fails to bootstrap and exits. The
# destroy is right. The exit leaves the cell unplaced, and nothing retries,
# so every further attempt is a hand-typed relaunch. B8 lost fifteen boxes
# that way. This loop takes the next offer instead, and it stops on the two
# conditions where retrying is wrong: the cell is placed, or the credit is
# gone.
#
# The bootstrap's GPU gate is what makes the retry cheap. A box that cannot
# count a CUDA device fails in about three minutes, for about two cents,
# and this loop moves to the next offer.
set -uo pipefail

CELL="${1:?usage: r2_box_retry.sh <cell> <stop> [stop...]}"
shift
STOPS="$*"
[ -n "$STOPS" ] || { echo "ABORT: no stops" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
TRIES="${BOX_TRIES:-6}"
# Below this the study cannot finish a leg, so a new box only burns what is
# left. The PR says so rather than the loop redefining the deliverable.
MIN_CREDIT="${MIN_CREDIT:-1.50}"
VASTRUN_DIR="${VASTRUN_DIR:-$(cd "$HERE" && git rev-parse --show-toplevel)}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [retry $CELL] $*" | tee -a "$RES/r2_boxes.log"; }

credit(){ (cd "$VASTRUN_DIR" && timeout 120 vastrun-balance 2>/dev/null) \
            | awk '/Credit/ {gsub(/\$/,"",$2); print $2; exit}'; }

for (( t = 1; t <= TRIES; t++ )); do
  c="$(credit)"
  if [ -n "$c" ] && awk -v c="$c" -v m="$MIN_CREDIT" 'BEGIN{exit !(c < m)}'; then
    say "STOP: credit \$$c is under \$$MIN_CREDIT — not renting another box"
    exit 9
  fi
  say "attempt $t/$TRIES (credit \$${c:-?})"
  if bash "$HERE/r2_launch_cell.sh" "$CELL" $STOPS; then
    say "placed on attempt $t"
    exit 0
  fi
  say "attempt $t failed; next offer in 30s"
  sleep 30
done

say "gave up after $TRIES attempts"
exit 1
