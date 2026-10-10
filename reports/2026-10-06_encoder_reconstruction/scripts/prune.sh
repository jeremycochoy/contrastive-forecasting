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
#   * its head training ended: its directory holds a `*_final.pth`, or its
#     job has a score file. No file of the directory changes again.
#   * for the `*_final.pth` itself: its job has a score file, so no score
#     reads it again.
# The rule reads the tree before it deletes a file, so the order of the
# manifest changes nothing.
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

has_score(){  # <tag>
  [ -n "$(find "$RES" -name "score_$1.txt" -size +0 2>/dev/null | head -1)" ]
}

# Pass 1: the files to delete, read off the unchanged tree.
doomed=()
kept=0
while IFS=$'\t' read -r bytes rel; do
  case "$rel" in ''|/*|*..*) continue ;; *.pth) ;; *) continue ;; esac
  f="$ROOT/$rel"
  [ -f "$f" ] || continue
  [ "$(stat -c %s "$f")" = "$bytes" ] || { kept=$(( kept + 1 )); continue; }
  dir=$(dirname "$f"); tag=$(basename "$dir")
  case "$f" in
    *_final.pth) has_score "$tag" || { kept=$(( kept + 1 )); continue; } ;;
    *) { compgen -G "$dir/*_final.pth" >/dev/null || has_score "$tag"; } \
         || { kept=$(( kept + 1 )); continue; } ;;
  esac
  doomed+=("$f")
done <"$MANIFEST"

# Pass 2: delete them.
freed=0
for f in "${doomed[@]}"; do
  size=$(stat -c %s "$f")
  if [ "$DRY" = "--dry-run" ]; then
    echo "would delete $f ($size bytes)"
  else
    rm -f "$f" && log "deleted ${f#"$ROOT"/} ($size bytes)"
  fi
  freed=$(( freed + size ))
done
log "done: ${#doomed[@]} file(s), $(( freed / 1048576 )) MiB${DRY:+ (dry run)}, $kept kept"
