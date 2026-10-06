#!/bin/bash
# #425 — delete the box copies of this card's checkpoints that elisa holds.
#
# The box has 16 GB free, and the heads of this card fill about 19 GB. So
# `sync_box.sh` copies each finished head to elisa, builds a manifest of what
# elisa holds there, and calls this script on the box with it.
#
# A box file under CF425_PRUNE_ROOT is deleted only when ALL of these hold:
#   * it is a `.pth` file. Logs, scores and per-config tables stay.
#   * the manifest names it, with the byte size it has on the box.
#   * its directory holds a `*_final.pth`: the head training ended, so no
#     file of the directory changes again.
#   * for the `*_final.pth` itself: its job has a score file, so no score
#     reads it again.
# Nothing outside CF425_PRUNE_ROOT is read or deleted: the input checkpoints
# of the queue stay on the box.
#
# Usage, on the box:  bash prune.sh <manifest> [--dry-run]
#   manifest lines: "<bytes>\t<path under CF425_PRUNE_ROOT>"
set -uo pipefail

MANIFEST="${1:?usage: prune.sh <manifest> [--dry-run]}"
DRY="${2:-}"
ROOT="${CF425_PRUNE_ROOT:-/workspace/ckpt/cf-425}"
RES="${CF425_RES:-/workspace/results/cf-425}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 prune] $*"; }

[ -f "$MANIFEST" ] || { log "ABORT: no manifest at $MANIFEST"; exit 2; }
[ -d "$ROOT" ] || { log "nothing under $ROOT"; exit 0; }

deleted=0; kept=0; freed=0
while IFS=$'\t' read -r bytes rel; do
  case "$rel" in ''|/*|*..*) continue ;; *.pth) ;; *) continue ;; esac
  f="$ROOT/$rel"
  [ -f "$f" ] || continue
  size=$(stat -c %s "$f")
  [ "$size" = "$bytes" ] || { kept=$(( kept + 1 )); continue; }
  dir=$(dirname "$f")
  compgen -G "$dir/*_final.pth" >/dev/null || { kept=$(( kept + 1 )); continue; }
  case "$f" in
    *_final.pth)
      tag=$(basename "$dir")
      if [ -z "$(find "$RES" -name "score_$tag.txt" -size +0 2>/dev/null | head -1)" ]; then
        kept=$(( kept + 1 )); continue
      fi ;;
  esac
  if [ "$DRY" = "--dry-run" ]; then
    echo "would delete $f ($size bytes)"
  else
    rm -f "$f" && log "deleted $rel ($size bytes)"
  fi
  deleted=$(( deleted + 1 )); freed=$(( freed + size ))
done <"$MANIFEST"
log "done: $deleted file(s), $(( freed / 1048576 )) MiB${DRY:+ (dry run)}, $kept kept"
