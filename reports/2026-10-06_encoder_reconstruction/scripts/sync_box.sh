#!/bin/bash
# #425 — bring this card's box files to elisa, then free the box copies that
# elisa holds. Run it on elisa, in a loop, for the whole queue.
#
# One tick:
#   1. The box results (/workspace/results/cf-425: scores, logs) go to
#      CF425_RESULTS_MIRROR.
#   2. Each head directory under /workspace/ckpt/cf-425 that holds a
#      `*_final.pth` goes to elisa's mirror of the box,
#      ~/checkpoints_backup/cf-412/vast_lr100x/cf-425: the head, its
#      optimizer, its losses and its score outputs. A directory with no
#      final head still trains, and its files still change, so it waits.
#   3. The manifest of the `.pth` files that elisa holds there goes to the
#      box, and `prune.sh` deletes the box copies of the same byte size.
#
# One tar stream per tick, into a staging directory. Each file takes its
# real name only at its listed size. A text file (log, CSV) that grew during
# the copy is kept at its larger size, and a checkpoint at another size
# waits for the next tick.
#
# Usage, on elisa:  bash sync_box.sh               # one tick
#                   bash sync_box.sh --loop 1800   # a tick every 30 minutes
set -uo pipefail

HOST="${CF425_HOST:-root@ssh5.vast.ai}"
PORT="${CF425_PORT:-31200}"
CK_ROOT="${CF425_BOX_ROOT:-/workspace/ckpt/cf-425}"
BOX_RES="${CF425_RES:-/workspace/results/cf-425}"
MIRROR="${CF425_MIRROR:-$HOME/checkpoints_backup/cf-412/vast_lr100x}/cf-425"
RES_MIRROR="${CF425_RESULTS_MIRROR:-$HOME/checkpoints_backup/cf-425/box_results}"
PRUNE="${CF425_PRUNE:-/workspace/cf-425/reports/2026-10-06_encoder_reconstruction/scripts/prune.sh}"
SSH=(ssh -p "$PORT" -o ConnectTimeout=20 -o ServerAliveInterval=30 "$HOST")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 sync] $*"; }

# "<bytes>\t<path>" of each file to bring from <box dir>. With "final", only
# the files of a directory (or of a subdirectory of one) that holds a
# `*_final.pth`.
box_listing(){  # <box dir> <all|final>
  "${SSH[@]}" "cd '$1' 2>/dev/null || exit 0
    find . -type f ! -path './locks/*' ! -name '*.tmp' -printf '%s\t%P\n' > /tmp/cf425_all.\$\$
    if [ '$2' = final ]; then
      find . -name '*_final.pth' -printf '%h\n' | sed 's|^\./||' | sort -u > /tmp/cf425_dirs.\$\$
      awk -F'\t' 'NR == FNR { d[\$1] = 1; next }
        { p = \$2; while (p ~ /\//) { sub(/\/[^\/]*\$/, \"\", p); if (p in d) { print; next } } }' \
        /tmp/cf425_dirs.\$\$ /tmp/cf425_all.\$\$
    else cat /tmp/cf425_all.\$\$; fi
    rm -f /tmp/cf425_all.\$\$ /tmp/cf425_dirs.\$\$" </dev/null
}

# The files of a listing that elisa lacks, or holds at another size.
wanted(){  # <listing> <local dir>
  while IFS=$'\t' read -r bytes rel; do
    [ -n "$rel" ] || continue
    [ "$(stat -c %s "$2/$rel" 2>/dev/null)" = "$bytes" ] || printf '%s\t%s\n' "$bytes" "$rel"
  done <<<"$1"
}

# One tar stream of the wanted files into a staging directory, then each
# file into place.
pull(){  # <box dir> <local dir> <wanted listing>
  local box="$1" dest="$2" want="$3" stage n=0 bytes rel got
  [ -n "$want" ] || return 0
  stage="$dest/.incoming.$$"
  mkdir -p "$stage"
  cut -f2 <<<"$want" | "${SSH[@]}" "cd '$box' && tar -cf - -T -" | tar -xf - -C "$stage" \
    || log "tar from $box ended with an error. Each file is checked below."
  while IFS=$'\t' read -r bytes rel; do
    got=$(stat -c %s "$stage/$rel" 2>/dev/null || echo 0)
    case "$rel" in
      *.pth) [ "$got" = "$bytes" ] || continue ;;
      *) [ "$got" -ge "$bytes" ] || continue ;;
    esac
    mkdir -p "$dest/$(dirname "$rel")"
    mv -f "$stage/$rel" "$dest/$rel" && n=$(( n + 1 ))
  done <<<"$want"
  rm -rf "$stage"
  log "$n of $(grep -c . <<<"$want") file(s) from $box"
}

tick(){
  local listing
  mkdir -p "$MIRROR" "$RES_MIRROR"
  listing="$(box_listing "$BOX_RES" all)"
  pull "$BOX_RES" "$RES_MIRROR" "$(wanted "$listing" "$RES_MIRROR")"
  listing="$(box_listing "$CK_ROOT" final)"
  pull "$CK_ROOT" "$MIRROR" "$(wanted "$listing" "$MIRROR")"
  # The manifest: what elisa holds, at its size.
  (cd "$MIRROR" && find . -name '*.pth' -type f -printf '%s\t%P\n') \
    | "${SSH[@]}" "cat > '$BOX_RES/elisa_manifest.tsv' && bash '$PRUNE' '$BOX_RES/elisa_manifest.tsv' ${CF425_PRUNE_DRY:+--dry-run}"
  "${SSH[@]}" "df -h /workspace | tail -1" </dev/null | sed 's/^/  box disk: /'
}

if [ "${1:-}" = "--loop" ]; then
  every="${2:-1800}"
  while :; do tick; sleep "$every"; done
fi
tick
