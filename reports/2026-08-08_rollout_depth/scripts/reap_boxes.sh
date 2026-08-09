#!/bin/bash
# #373 — destroy a box once its queue has drained AND its work is here.
#
# Usage: bash reap_boxes.sh            # one pass
#
# A drained box bills at $0.36-$0.48/h for nothing, and this study's ceiling
# is $7.31 of credit. But a box destroyed before its last checkpoint lands
# takes the run with it, so the gate is not "the queue is done" — it is
# "every 40k checkpoint the remote holds is HERE, at the same size".
#
# Only boxes in results/boxes.tsv are ever touched, and each is destroyed
# with the label this session provisioned it under, which `vastrun-destroy`
# checks against the on-instance marker. The vast.ai account is shared with
# other agent sessions (CLAUDE.md): a box not in this table is somebody
# else's, whatever it is labelled.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
BOXES="${BOXES_FILE:-$STUDY/results/boxes.tsv}"
SYNC_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
LABEL="${CF373_LABEL:-cf373-rollout-a}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=10 -n)
[ -f "$BOXES" ] || { echo "ABORT: no box table at $BOXES" >&2; exit 2; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [reap] $*" | tee -a "$STUDY/results/reap.log"; }

while IFS=$'\t' read -r lbl id host port jobs; do
  case "$lbl" in ''|'#'*) continue ;; esac

  state=$(timeout 45 ssh "${SSH_OPTS[@]}" -p "$port" "root@$host" '
      test -f /root/cf/reports/2026-08-08_rollout_depth/results/QUEUE_DONE \
        && echo DONE || echo BUSY
      find /root/cf373_runs -name "*_40k.pth" ! -name "*_optimizer.pth" \
        -printf "CK %s %p\n" 2>/dev/null
      find /root/cf373_runs -name "*_40k_optimizer.pth" \
        -printf "CK %s %p\n" 2>/dev/null
    ' 2>/dev/null)
  [ -n "$state" ] || { say "$lbl unreachable — leaving it"; continue; }
  grep -q '^DONE$' <<<"$state" || continue

  missing=0 n=0
  while read -r _tag size path; do
    [ -n "${path:-}" ] || continue
    n=$(( n + 1 ))
    dst="$SYNC_BASE/$lbl/sync/${path#/root/cf373_runs/}"
    if [ ! -f "$dst" ] || [ "$(wc -c <"$dst")" != "$size" ]; then
      say "$lbl NOT YET: $(basename "$path") is not here at $size B"
      missing=$(( missing + 1 ))
    fi
  done < <(grep '^CK ' <<<"$state")

  if [ "$n" -eq 0 ]; then
    say "$lbl drained but holds no 40k checkpoint — leaving it for inspection"
    continue
  fi
  [ "$missing" -eq 0 ] || continue

  say "$lbl drained, all $n checkpoint file(s) verified local — destroying $id"
  timeout 300 vastrun-destroy "$id" "$LABEL" 2>&1 | sed 's/^/  /' \
    | tee -a "$STUDY/results/reap.log"
  # Stop its sync loop: the box is gone and the loop would poll a dead host.
  for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$SYNC_BASE/$lbl" ] && kill "$p"
  done
  # Drop the row, so a second pass does not try again.
  tmp="$BOXES.tmp.$$"
  grep -v -P "^$lbl\t" "$BOXES" > "$tmp" && mv -f "$tmp" "$BOXES"
done < "$BOXES"
