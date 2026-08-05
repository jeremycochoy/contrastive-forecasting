#!/bin/bash
# Resolve the backbone checkpoint to evaluate at a named training step.
#
# Usage: resolve_eval_checkpoint.sh <runs-dir> <run-name> <step-k> [<path>]
#
# Prints the resolved path on stdout, and a `[checkpoint] …` provenance line
# on stderr so the caller's log records which file produced its numbers.
#
# A resumed run keeps its own `_r<N>` safe-run name, so one (run name, step)
# pair can leave several backbones on disk — `NAME_40k.pth` from the first
# attempt and `NAME_r3_40k.pth` from the resume. They are different models.
# Picking between them by modification time evaluates one replicate under the
# other's cell name and records nothing (#390), so this aborts instead and
# lists what matched. Pass <path> to name the replicate you mean: it must be
# `<run-name>_<step-k>k.pth` or `<run-name>_r<N>_<step-k>k.pth`, since it
# selects a replicate of this (run name, step) pair rather than relabelling
# some other checkpoint as one.
#
# This is eval-time selection at a *named* step. Resume logic that wants the
# newest checkpoint of a live run is a different question; don't use this.
#
# The caller names its output cell after the replicate this returns, using
# `eval_cell_identity.sh` — the same file this reads the name grammar from.
#
# Exit codes: 2 usage or a missing eval_cell_identity.sh, 3 no candidate,
#             4 named path missing, 5 ambiguous, 6 named path is not this
#             run's checkpoint at this step.
set -uo pipefail

SELF="$(basename "$0")"

# The snapshot-name grammar lives next door, because the output cell is named
# from the same parse: a second copy of `(_r[0-9]+)?` here would drift from
# the one the cell name is built with, and a drifted copy is how a checkpoint
# this accepts gets filed under a cell name it does not belong to.
# `cd -P`: this file is reached through a `scripts/` symlink too.
IDENTITY="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_cell_identity.sh"
[ -f "$IDENTITY" ] || {
  echo "ABORT: $SELF cannot find eval_cell_identity.sh at $IDENTITY" >&2
  exit 2; }
# shellcheck source=eval_cell_identity.sh
source "$IDENTITY"

[ $# -ge 3 ] && [ $# -le 4 ] || {
  echo "usage: $SELF <runs-dir> <run-name> <step-k> [<path>]" >&2; exit 2; }

RUNS="$1"; NAME="$2"; STEP_K="$3"; EXPLICIT="${4-}"

# --- Caller named the file: honour it, verify it, say so. ------------------
# Verify is both halves. The file must exist, and its name must say it belongs
# to this (run name, step) pair: callers name their output cell after (arm,
# step) whatever this path points at, so a file from another step or another
# run would publish one backbone's numbers under another's name — the failure
# this script exists to stop, walking in through the door it opens. The
# override picks a replicate; it does not rename one.
if [ -n "$EXPLICIT" ]; then
  [ -f "$EXPLICIT" ] || {
    echo "ABORT: named checkpoint does not exist: $EXPLICIT" >&2; exit 4; }
  if ! ckpt_is_run_step "$NAME" "$STEP_K" "$EXPLICIT"
  then
    {
      echo "ABORT: named checkpoint is not '$NAME' at step ${STEP_K}k:"
      echo "  $EXPLICIT"
      echo "Its name must be '${NAME}_${STEP_K}k.pth', or"
      echo "'${NAME}_r<N>_${STEP_K}k.pth' for a resumed replicate. Whatever"
      echo "this file is, the caller files the result under ($NAME, ${STEP_K}k),"
      echo "so any other name publishes one backbone under another's."
    } >&2
    exit 6
  fi
  echo "[checkpoint] $NAME @ ${STEP_K}k -> $EXPLICIT (explicit override)" >&2
  echo "$EXPLICIT"
  exit 0
fi

# --- Otherwise: the base run and every `_r<N>` resume at this step. --------
# The glob narrows and `ckpt_is_run_step` — the same predicate the override
# above is checked with — decides. `_r[0-9]*_` and not `_r*_` keeps a sibling
# run whose suffix starts with an `r` (`_revin_`) out of the gather; the
# predicate then drops what a glob cannot express, `_r3x`, which would
# otherwise be returned as a replicate and refused one stage later by the
# caller naming its cell from it.
CANDIDATES=()
for f in "$RUNS/${NAME}_${STEP_K}k.pth" "$RUNS/${NAME}"_r[0-9]*_${STEP_K}k.pth; do
  [ -f "$f" ] || continue
  ckpt_is_run_step "$NAME" "$STEP_K" "$f" || continue
  CANDIDATES+=("$f")
done

if [ ${#CANDIDATES[@]} -eq 0 ]; then
  echo "ABORT: no ${STEP_K}k checkpoint for run '$NAME' under $RUNS" >&2
  exit 3
fi

if [ ${#CANDIDATES[@]} -gt 1 ]; then
  {
    echo "ABORT: ${#CANDIDATES[@]} checkpoints match run '$NAME' at step ${STEP_K}k:"
    printf '  %s\n' "${CANDIDATES[@]}"
    echo "A resume writes its own '_r<N>' run name, so these are different"
    echo "backbones. Modification time does not say which one a result should"
    echo "cite. Name the one you mean:"
    echo "  $SELF <runs-dir> <run-name> <step-k> <path>"
  } >&2
  exit 5
fi

echo "[checkpoint] $NAME @ ${STEP_K}k -> ${CANDIDATES[0]}" >&2
echo "${CANDIDATES[0]}"
