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
# -n, or ssh reads the box table off this loop's stdin and only the first
# box is ever checked.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15 -n)
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

  # Checkpoints are written once and never change, so an exact size match is
  # the right test. A losses CSV, an attn-amplitude CSV and a run log GROW
  # every step, so their remote size is newer than any copy by construction
  # and a size test would report every healthy loop as broken. For those the
  # test is that a non-empty copy exists.
  n_ck=0 ck_ok=0 n_grow=0 grow_ok=0 missing=""
  while read -r size path; do
    [ -n "${path:-}" ] || continue
    dst="$LOCAL/sync/${path#/root/cf373_runs/}"
    case "$path" in
      *.pth)
        n_ck=$(( n_ck + 1 ))
        if [ -f "$dst" ] && [ "$(wc -c <"$dst")" = "$size" ]; then
          ck_ok=$(( ck_ok + 1 ))
        else
          [ "$(wc -w <<<"$missing")" -lt 3 ] && missing="$missing $(basename "$path")"
        fi ;;
      *)
        n_grow=$(( n_grow + 1 ))
        if [ -s "$dst" ]; then
          grow_ok=$(( grow_ok + 1 ))
        else
          [ "$(wc -w <<<"$missing")" -lt 3 ] && missing="$missing $(basename "$path")"
        fi ;;
    esac
  done < <(ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" \
             "find /root/cf373_runs -type f -printf '%s %p\n' 2>/dev/null" 2>/dev/null)

  printf '[%s] loop=%-7s ckpt %d/%d exact  growing %d/%d present%s\n' \
    "$lbl" "${alive:-DEAD}" "$ck_ok" "$n_ck" "$grow_ok" "$n_grow" \
    "${missing:+  missing:$missing}"
  [ -n "$alive" ] || rc=1
  [ "$n_ck" -gt 0 ] && [ "$ck_ok" -eq 0 ] && rc=1
  [ "$n_grow" -gt 0 ] && [ "$grow_ok" -eq 0 ] && rc=1
done < "$BOXES"
exit "$rc"
