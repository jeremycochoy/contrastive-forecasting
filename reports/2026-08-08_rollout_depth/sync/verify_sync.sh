#!/bin/bash
# #373 — verify a sync loop by `ls`, not by reading its log.
#
# Usage: bash sync/verify_sync.sh [label ...]
#
# CLAUDE.md § Remote Machine Monitoring: a sync log with no `✗` line can
# mean the pull pattern never matched anything, which looks exactly like a
# loop that is running and finding nothing. So this compares the remote
# listing with the local tree, file by file, by name and size.
#
# Prints one line per box: how many remote files exist, how many are here at
# the right size, and the first three that are not. Exits non-zero if any
# box is missing a file the remote has held for more than one tick.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
BOXES="${BOXES_FILE:-$STUDY/results/boxes.tsv}"
LOCAL_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)
[ -f "$BOXES" ] || { echo "ABORT: no box table at $BOXES" >&2; exit 2; }
WANT="${*:-}"
rc=0

while IFS=$'\t' read -r lbl id host port jobs; do
  case "$lbl" in ''|'#'*) continue ;; esac
  if [ -n "$WANT" ]; then
    case " $WANT " in *" $lbl "*) ;; *) continue ;; esac
  fi
  LOCAL="$LOCAL_BASE/$lbl"

  alive=""
  for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$LOCAL" ] && { alive="$p"; break; }
  done

  n_remote=0 n_ok=0 missing=""
  while read -r size path; do
    [ -n "${path:-}" ] || continue
    n_remote=$(( n_remote + 1 ))
    rel="${path#/root/cf373_runs/}"
    dst="$LOCAL/sync/$rel"
    if [ -f "$dst" ] && [ "$(wc -c <"$dst")" = "$size" ]; then
      n_ok=$(( n_ok + 1 ))
    else
      [ "$(wc -w <<<"$missing")" -lt 3 ] && missing="$missing $(basename "$path")"
    fi
  done < <(ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
             "find /root/cf373_runs -type f -printf '%s %p\n' 2>/dev/null" 2>/dev/null)

  printf '[%s] loop=%s remote=%d local_ok=%d%s\n' \
    "$lbl" "${alive:-DEAD}" "$n_remote" "$n_ok" \
    "${missing:+  missing:$missing}"
  [ -n "$alive" ] || rc=1
  [ "$n_remote" -gt 0 ] && [ "$n_ok" -eq 0 ] && rc=1
done < "$BOXES"
exit "$rc"
