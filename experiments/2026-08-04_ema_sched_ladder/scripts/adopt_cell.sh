#!/bin/bash
# #393 — finish a rented box's stop here on elisa, so the box can go back.
#
# Usage:  bash scripts/adopt_cell.sh <cell> <stop_steps> <student|teacher> [gpu]
#
# Why this exists.
#
# A stop costs one head training on a GPU (30,000 steps, ~1.4 h) and one
# GIFT-Eval. Since PR #394 the eval runs on elisa, so a box that has trained
# its head has nothing left to do and still bills $0.36-$0.49 an hour for
# the hours the eval takes. Worse on boxes E and F: their drivers were
# launched before `stop_scores()` learned to run the two heads
# concurrently, so their teacher head does not even START until the
# student's eval returns. That is a second idle hour and a half each,
# bought at the rented rate, for work this machine can do for nothing.
#
# The backbone is already here. sync_loop.sh has been pulling every
# periodic checkpoint and its optimizer companion since the run began, so
# `<sync root>/sync/<cell>/leg_<N>k/cf393_<cell>_<N>k.pth` is a byte-for-byte
# copy of what the box holds. This stages it under the local runs root in
# the layout `eval_stop.sh` expects and then calls `eval_stop.sh` itself —
# same head hyper-parameters, same seed, same encoder marker, same 97-config
# eval, same score file — so an adopted stop and a native one are the same
# measurement, produced by the same script.
#
# The copy is verified against the box's own copy where one is reachable:
# the broker pulled that stop's backbone for the other head's eval, and if
# the two differ in size this refuses rather than evaluating a checkpoint
# that is not the one the ladder trained.
#
# RELEASE THE BOX FIRST, or at least before its own driver reaches the same
# head. Two machines training one (cell, stop, head) produce two scores for
# one key, which merge_pooled.sh reports as a collision needing a human.
set -uo pipefail

CELL="${1:?usage: adopt_cell.sh <cell> <stop_steps> <student|teacher> [gpu]}"
STOP="${2:?stop steps}"
ENC="${3:?student|teacher}"
GPU="${4:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WT="${WT:-$(dirname "$(dirname "$EXP")")}"
. "$HERE/leg_paths.sh"
ROOT="$(runs_root)" || exit 2
STOP_K=$(( STOP / 1000 ))
NAME="cf393_${CELL}"
DEST="$ROOT/$CELL/leg_${STOP_K}k"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [adopt] $*"; }

# Where sync_loop.sh puts this box's tree. One directory per box label.
find_source() {  # prints the newest matching backbone under any sync root
  local best="" f
  for f in "$HOME"/cf393_sync*/2026-08-04_ema_sched_ladder/sync/"$CELL"/leg_${STOP_K}k/"${NAME}_${STOP_K}k.pth"; do
    [ -s "$f" ] || continue
    best="$f"
  done
  printf '%s' "$best"
}

SRC="$(find_source)"
[ -n "$SRC" ] || { say "ABORT: no ${NAME}_${STOP_K}k.pth under any ~/cf393_sync*/ tree"; exit 3; }

# The box's own copy of the same file, if the broker fetched it for the
# other head. Same bytes or this is not the ladder's checkpoint.
for peer in "$ROOT"/_broker/*/"$CELL"/bb${STOP_K}k_*/backbone.pth; do
  [ -s "$peer" ] || continue
  a=$(stat -c %s "$SRC"); b=$(stat -c %s "$peer")
  if [ "$a" != "$b" ]; then
    say "ABORT: $SRC is $a bytes, the box's copy $peer is $b — refusing"
    exit 4
  fi
  say "size check ok against $peer ($a bytes)"
  break
done

mkdir -p "$DEST" || exit 2
if [ ! -s "$DEST/${NAME}_${STOP_K}k.pth" ]; then
  cp -a "$SRC" "$DEST/${NAME}_${STOP_K}k.pth.tmp" \
    && mv -f "$DEST/${NAME}_${STOP_K}k.pth.tmp" "$DEST/${NAME}_${STOP_K}k.pth" \
    || { say "ABORT: staging the backbone failed"; exit 5; }
  say "staged $(basename "$SRC") -> $DEST"
else
  say "backbone already staged at $DEST"
fi
# The optimizer is not needed to score a stop, but a cell adopted here can
# only be EXTENDED from here, and an extension without the optimizer loses
# the step counter, the RNG state and AdamW's momentum (CLAUDE.md).
if [ -s "${SRC%.pth}_optimizer.pth" ] && [ ! -s "$DEST/${NAME}_${STOP_K}k_optimizer.pth" ]; then
  cp -a "${SRC%.pth}_optimizer.pth" "$DEST/${NAME}_${STOP_K}k_optimizer.pth.tmp" \
    && mv -f "$DEST/${NAME}_${STOP_K}k_optimizer.pth.tmp" \
             "$DEST/${NAME}_${STOP_K}k_optimizer.pth"
  say "staged the optimizer companion too"
fi

SCORE="$ROOT/$CELL/eval/score_bb${STOP_K}k_${ENC}.txt"
if [ -s "$SCORE" ]; then
  say "$CELL bb${STOP_K}k $ENC already scored: $(cat "$SCORE")"
  exit 0
fi

HEAD_STEPS=$(( STOP < 100000 ? 15000 : 30000 ))
say "$CELL bb${STOP_K}k $ENC — head ${HEAD_STEPS} steps on GPU $GPU, then GIFT-Eval here"
exec env WT="$WT" BB_GPU="$GPU" bash "$HERE/eval_stop.sh" \
     "$CELL" "$STOP" "$ENC" "$HEAD_STEPS" "$SCORE"
