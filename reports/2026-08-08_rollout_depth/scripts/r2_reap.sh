#!/bin/bash
# #373 round 2 — destroy a box once its cell's artefacts are here, byte for
# byte, and not before.
#
# Usage: bash r2_reap.sh [poll seconds]
#
# The gate is `ls`, not a log line. For every file the box holds under
# /root/cf373_runs, the local copy must exist at exactly the remote size.
# Round 1 lost nothing to this loop and paid $0.31 for a box that drained at
# 04:15 and was found at 05:06, so the poll is short.
#
# Only boxes this session's own launcher recorded in r2_boxes.tsv are ever
# touched, and only by the id in that row. The vast.ai account is shared
# with other agent sessions.
set -uo pipefail

POLL="${1:-300}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
BOXES="$RES/r2_boxes.tsv"
SYNC_BASE="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"
VASTRUN_DIR="${VASTRUN_DIR:-$(cd "$HERE" && git rev-parse --show-toplevel)}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [reap] $*" | tee -a "$RES/r2_reap.log"; }

log "start poll=${POLL}s"
while :; do
  [ -f "$BOXES" ] || { sleep "$POLL"; continue; }
  while IFS=$'\t' read -r cell id host port label stops; do
    [ -n "${cell:-}" ] || continue
    case "$cell" in \#*) continue;; esac
    [ -f "$RES/r2_reaped_$cell" ] && continue

    # Held open on purpose: `touch results/HOLD_<cell>` to keep a box alive.
    [ -f "$RES/HOLD_$cell" ] && { log "$cell held by results/HOLD_$cell"; continue; }

    done_marker=$(ssh "${SSH_OPTS[@]}" -n -p "$port" "root@$host" \
      "test -f /root/cf/reports/2026-08-08_rollout_depth/results/CELL_${cell}_DONE && echo yes" 2>/dev/null)
    [ "$done_marker" = "yes" ] || continue

    # Every remote artefact, at the remote size, in this cell's local tree.
    missing=0; nfiles=0
    while read -r size path; do
      [ -n "${path:-}" ] || continue
      rel="${path#/root/cf373_runs/}"
      dst="$SYNC_BASE/$cell/sync/$rel"
      nfiles=$(( nfiles + 1 ))
      if [ ! -f "$dst" ] || [ "$(wc -c <"$dst")" != "$size" ]; then
        log "$cell NOT YET: $rel is not here at $size B"; missing=1
      fi
    done < <(ssh "${SSH_OPTS[@]}" -n -p "$port" "root@$host" \
             "find /root/cf373_runs -type f -printf '%s %p\n' 2>/dev/null" 2>/dev/null)
    [ "$nfiles" -gt 0 ] || { log "$cell reports DONE but lists no artefacts — leaving it"; continue; }
    [ "$missing" -eq 0 ] || continue

    log "$cell drained, all $nfiles file(s) verified local — destroying $id ($label)"
    out=$( (cd "$VASTRUN_DIR" && timeout 300 vastrun-destroy "$id" "$label") 2>&1 )
    if grep -qi "no marker" <<<"$out"; then
      # The kit writes its ownership marker AFTER its SSH check, so a box
      # adopted by provision_box.sh's probe carries none. This row was
      # written by this session's own launcher and its artefacts are all
      # here, so --force is the only thing left.
      log "$cell: no vastrun marker (adopted box) — destroying with --force"
      out=$( (cd "$VASTRUN_DIR" && timeout 300 vastrun-destroy "$id" --force) 2>&1 )
    fi
    printf '%s\n' "$out" | sed 's/^/    /' | tee -a "$RES/r2_reap.log" >/dev/null
    touch "$RES/r2_reaped_$cell"
    log "$cell reaped"
  done < "$BOXES"
  sleep "$POLL"
done
