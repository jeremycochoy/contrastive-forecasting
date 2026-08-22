#!/bin/bash
# #404 — copy one box's backbone artefacts into the CANONICAL tree.
#
# WHY THIS EXISTS. Every figure, table and eval of this card reads ONE tree,
# `~/cf404_sync/box_a/sync`. A box that trains an arm writes its own tree, and
# the driver's stage 7 copies the result across. Round 4's driver was stopped
# before its stage 7, so `s08c` and `s08d` reached 40,000 steps and their
# artefacts stayed under `box_r4`. To the canonical tree those two arms look
# untrained.
#
# This copies a backbone leg and nothing else. It never copies a head or an
# eval: `s08c` and `s08d` have none, on purpose.
#
# It is idempotent and it never shrinks a file. A destination that already
# holds the same bytes or more is kept, so a re-run after a fuller sync cannot
# replace a complete file with a partial one.
#
# Usage:  bash scripts/promote_backbones.sh <from-label> <arm> [<arm>...]
#         bash scripts/promote_backbones.sh box_r4 s08c s08d
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

FROM_LABEL="${1:?usage: promote_backbones.sh <from-label> <arm>...}"; shift
[ "$#" -ge 1 ] || { echo "ABORT: name at least one arm" >&2; exit 2; }

FROM_DIR="$HOME/cf404_sync/$FROM_LABEL"
FROM="$FROM_DIR/sync"
TO_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
TO="$TO_DIR/sync"
STOP="${STOP:-$CF404_STOPS}"
KK=$(( STOP / 1000 ))

[ -d "$FROM" ] || { echo "ABORT: no tree at $FROM" >&2; exit 2; }
say(){ echo "[#404 promote] $*"; }
say "from $FROM"
say "to   $TO"

copy(){  # <src> <dst>
  local src="$1" dst="$2" sb db
  [ -f "$src" ] || { say "  MISSING $(basename "$src")"; return 1; }
  sb="$(wc -c <"$src")"
  if [ -f "$dst" ]; then
    db="$(wc -c <"$dst")"
    if [ "$db" -ge "$sb" ]; then
      say "  keep $(basename "$dst") ($db B, source $sb B)"; return 0
    fi
  fi
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst" || return 1
  say "  $(basename "$dst") $sb B"
}

rc=0
for arm in "$@"; do
  cf404_require_arm "$arm" || { rc=1; continue; }
  name="$(cf404_run_name "$arm")"
  say "$arm"
  src_leg="$FROM/$arm/$CF404_CELL/leg_${KK}k"
  dst_leg="$TO/$arm/$CF404_CELL/leg_${KK}k"
  for f in "${name}_${KK}k.pth" "${name}_${KK}k_optimizer.pth" \
           "${name}_losses.csv" "${name}_attn_amplitude.csv" \
           "${name}_latent_drift.csv"; do
    copy "$src_leg/$f" "$dst_leg/$f" || rc=1
  done
  copy "$FROM_DIR/results/run_${name}.log" "$TO_DIR/results/run_${name}.log" \
    || rc=1
done

say "done, rc=$rc"
exit "$rc"
