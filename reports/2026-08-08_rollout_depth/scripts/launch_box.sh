#!/bin/bash
# #373 — give one bootstrapped box its jobs, and start watching it.
#
# Usage: bash launch_box.sh <label> <instance id> <host> <port> <job> [job...]
#        job = <cell>:<k>:<steps>, e.g. B5:0:40000
#
# Records the box in results/boxes.tsv (the table launch_sync.sh and
# verify_sync.sh both read), starts queue_remote.sh detached on the box,
# then starts this box's sync loop and waits for its first tick to land.
#
# The queue is detached with setsid: an SSH drop must not take the training
# run with it, and on a rented box an SSH drop is routine.
set -uo pipefail

LBL="${1:?usage: launch_box.sh <label> <id> <host> <port> <job>...}"
ID="${2:?instance id}"
HOST="${3:?ssh host}"
PORT="${4:?ssh port}"
shift 4
[ $# -gt 0 ] || { echo "ABORT: no jobs for $LBL" >&2; exit 2; }
JOBS="$*"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
BOXES="$RES/boxes.tsv"
mkdir -p "$RES"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15)
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [box $LBL] $*" | tee -a "$RES/boxes.log"; }
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@"; }

rsh 'test -f /root/bootstrap.log && grep -q BOOTSTRAP_OK /root/bootstrap.log' \
  || { say "ABORT: box is not bootstrapped"; exit 3; }

say "jobs: $JOBS"
# The subshell, the three redirections and the bare `exit 0` are all load
# bearing. ssh holds the session open until every descriptor on the channel
# closes, so a backgrounded remote command that inherits ANY of them — an
# `echo queued` on stdout is enough — hangs the caller forever while the job
# it started runs on happily behind it.
rsh "(setsid bash /root/cf/reports/2026-08-08_rollout_depth/scripts/queue_remote.sh \
      $JOBS </dev/null >/root/queue.log 2>&1 &) ; exit 0" >/dev/null 2>&1 \
  || { say "ABORT: queue"; exit 4; }

# The table is the single source of truth for host/port. Rewrite this
# label's row rather than append, so a re-provisioned box does not leave a
# stale address that a sync loop then polls forever.
tmp="$BOXES.tmp.$$"
{ [ -f "$BOXES" ] && grep -v -P "^$LBL\t" "$BOXES"; \
  printf '%s\t%s\t%s\t%s\t%s\n' "$LBL" "$ID" "$HOST" "$PORT" "$JOBS"; } \
  | sort > "$tmp" && mv -f "$tmp" "$BOXES"

bash "$STUDY/sync/launch_sync.sh" "$LBL"
say "launched — verify the first tick with sync/verify_sync.sh $LBL"
