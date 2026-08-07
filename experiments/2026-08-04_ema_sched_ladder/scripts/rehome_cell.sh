#!/bin/bash
# #393 — move a released box's cell onto elisa so its ladder can carry on.
#
# Usage:  bash scripts/rehome_cell.sh <cell> [<box label>]
#         DRY=1 bash scripts/rehome_cell.sh <cell>     # report, change nothing
#
# A cell whose box was given back is recorded `budget_stop`: it stopped
# because the spend order ended it, not because the extend rule did. That
# is a real stop for a cell the rule also stopped — but arm5_combab_alignT
# read `both_down` at bb100k, so the rule wanted another 100k and the
# credit is what said no. Elisa's GPUs are free once phase 1's heads clear
# and they cost nothing, so those cells can climb again here.
#
# Rehoming is bookkeeping, not computation. ladder.py replays a cell from
# step 0 and skips what is already on disk, so all four of these have to be
# in the places elisa's own scripts look, or the replay retrains work that
# has already been paid for:
#
#   1. every leg checkpoint and its optimizer companion, under the runs
#      root — without the optimizer an extension loses the step counter,
#      the RNG state and AdamW's momentum (CLAUDE.md);
#   2. the score files, so eval_stop.sh skips a stop already scored rather
#      than training a 30,000-step head for a number we have;
#   3. the ladder rows, so stop_scores() has `recorded` and the extend rule
#      can be re-derived from the same scores the box derived it from;
#   4. the claim, so run_leg.sh stops refusing the cell as another
#      machine's. That guard is the reason two copies of one cell have
#      never written the same filenames into two roots; it is moved, not
#      removed.
#
# Refuses while any driver for the cell is alive. Two drivers on one cell
# is the failure the claims file exists to prevent.
set -uo pipefail

CELL="${1:?usage: rehome_cell.sh <cell> [box label]}"
LBL="${2:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
. "$HERE/leg_paths.sh"
ROOT="$(runs_root)" || exit 2
NAME="cf393_$CELL"
ME="$(tr -dc 'a-zA-Z0-9_-' <"$RES/MACHINE" 2>/dev/null)"
[ -n "$ME" ] || { echo "ABORT: no $RES/MACHINE"; exit 2; }

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [rehome] $*"; }
do_it(){ [ -z "${DRY:-}" ]; }

grep -qE "^${CELL}\b" <<<"$(awk '{print $1}' "$RES/cell_claims.txt")" \
  || say "note: $CELL is not listed in cell_claims.txt"

if pgrep -f "[l]adder\.py --cells.*$CELL" >/dev/null 2>&1; then
  say "ABORT: a driver for $CELL is running here — rehoming under it would fork the cell"
  exit 3
fi

# 1. checkpoints ------------------------------------------------------------
staged=0
for src in "$HOME"/cf393_sync*/2026-08-04_ema_sched_ladder/sync/"$CELL"/leg_*/"$NAME"_[0-9]*k.pth; do
  [ -s "$src" ] || continue
  leg="$(basename "$(dirname "$src")")"
  dst="$ROOT/$CELL/$leg/$(basename "$src")"
  [ -s "$dst" ] && continue
  if do_it; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst" || { say "ABORT: copying $src failed"; exit 4; }
    [ -s "${src%.pth}_optimizer.pth" ] && {
      cp -a "${src%.pth}_optimizer.pth" "${dst%.pth}_optimizer.pth.tmp" \
        && mv -f "${dst%.pth}_optimizer.pth.tmp" "${dst%.pth}_optimizer.pth"; }
  fi
  say "checkpoint $leg/$(basename "$src")$(do_it || echo '  (dry)')"
  staged=$(( staged + 1 ))
done
[ "$staged" -gt 0 ] || say "no new checkpoint to stage (already here, or none synced)"

# An extension resumes from the FURTHEST checkpoint, so that one must have
# its optimizer or the resume is not a resume.
newest="$(newest_ckpt "$ROOT/$CELL" "$NAME")"
if [ -n "$newest" ] && [ ! -s "${newest%.pth}_optimizer.pth" ]; then
  say "ABORT: $(basename "$newest") has no optimizer companion here"
  exit 5
fi
[ -n "$newest" ] && say "furthest checkpoint: $(basename "$newest") (+ optimizer)"

# 2. score files ------------------------------------------------------------
# The broker's working copy is the durable one for a box's cell: it holds
# the score and the 97-config table that eval ran here to produce.
for wdir in "$ROOT/_broker"/${LBL:-*}/"$CELL"/bb*k_*; do
  [ -s "$wdir/score.txt" ] || continue
  stop="$(basename "$wdir")"
  dst="$ROOT/$CELL/eval/score_${stop}.txt"
  [ -s "$dst" ] && continue
  if do_it; then
    mkdir -p "$ROOT/$CELL/eval/$stop"
    cp -a "$wdir/score.txt" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
    [ -d "$wdir/gift" ] && cp -an "$wdir/gift" "$ROOT/$CELL/eval/$stop/" 2>/dev/null
  fi
  say "score $stop = $(cat "$wdir/score.txt")$(do_it || echo '  (dry)')"
done

# 3. ladder and decision rows ----------------------------------------------
# Appended to elisa's own CSVs, which live drivers are also appending to.
# Append is the only safe operation on those files; nothing here rewrites
# them. Rows already present are not added twice.
adopt_rows(){  # <basename> <local csv>
  local base="$1" out="$2" n=0 line key
  for f in "$RES/per_machine/${base}_"*.csv; do
    [ -f "$f" ] || continue
    while IFS= read -r line; do
      line="${line%$'\r'}"
      case "$line" in "$CELL",*) ;; *) continue ;; esac
      grep -qFx "$line" <(tr -d '\r' <"$out") 2>/dev/null && continue
      do_it && printf '%s\n' "$line" >> "$out"
      n=$(( n + 1 ))
    done < <(tail -n +2 "$f")
  done
  say "$base: $n row(s) adopted into $(basename "$out")$(do_it || echo '  (dry)')"
}
[ -f "$RES/ladder.csv" ]    && adopt_rows ladder    "$RES/ladder.csv"
[ -f "$RES/decisions.csv" ] && adopt_rows decisions "$RES/decisions.csv"

# 4. the claim --------------------------------------------------------------
old="$(awk -v c="$CELL" '$1==c {print $2}' "$RES/cell_claims.txt" | head -1)"
if [ "$old" = "$ME" ]; then
  say "claim already $ME"
elif do_it; then
  sed -i -E "s|^(${CELL}[[:space:]]+)[A-Za-z0-9_-]+|\1${ME}|" "$RES/cell_claims.txt"
  say "claim ${old:-none} -> $ME"
else
  say "claim ${old:-none} -> $ME  (dry)"
fi

say "$CELL is rehomed on $ME; a driver started here will replay it and extend from $(basename "${newest:-nothing}")"
