#!/bin/bash
# #393 — push the ladder's scripts to a remote box without corrupting a
# run that is using them.
#
# Usage:  bash scripts/deploy_scripts.sh <host> <port> [file ...]
#
# Why this is not just `scp`: scp writes THROUGH the existing inode
# (truncate, then write). bash reads a script lazily, holding a byte offset
# into that same inode, so replacing the file under a running script makes
# it resume at an offset that now lands in different text. On 2026-08-05
# that turned a live `eval_stop.sh` into
#
#     eval_stop.sh: line 97: ad-num-layers: command not found
#
# mid-word inside `--head-num-layers`, killing arm5_combab_alignS's driver
# after its 15,000-step head had already trained.
#
# Uploading to a temporary name and `mv`-ing over the target swaps the
# directory entry instead: a running bash keeps its descriptor on the old
# inode and finishes on the bytes it started with, while the next process
# to start picks up the new file. The local Edit tool already behaves this
# way, which is why elisa's cells came through the same change untouched.
set -uo pipefail

HOST="${1:?usage: deploy_scripts.sh <host> <port> [file ...]}"
PORT="${2:?usage: deploy_scripts.sh <host> <port> [file ...]}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE=/root/cf/experiments/2026-08-04_ema_sched_ladder/scripts
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20)

FILES=("$@")
[ ${#FILES[@]} -eq 0 ] && FILES=(gpu_gate.sh run_leg.sh eval_stop.sh ladder.py)

for f in "${FILES[@]}"; do
  [ -f "$HERE/$f" ] || { echo "ABORT: no $HERE/$f" >&2; exit 2; }
  scp "${SSH_OPTS[@]}" -P "$PORT" "$HERE/$f" "root@$HOST:$REMOTE/.$f.incoming" \
    >/dev/null 2>&1 || { echo "ABORT: upload of $f failed" >&2; exit 3; }
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "mv -f '$REMOTE/.$f.incoming' '$REMOTE/$f' && chmod +x '$REMOTE/$f'" \
    >/dev/null 2>&1 || { echo "ABORT: swap of $f failed" >&2; exit 4; }
  echo "  $f -> $HOST:$PORT"
done

# __pycache__ outlives a ladder.py swap and would shadow the new module.
ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "rm -rf '$REMOTE/__pycache__'" \
  >/dev/null 2>&1
