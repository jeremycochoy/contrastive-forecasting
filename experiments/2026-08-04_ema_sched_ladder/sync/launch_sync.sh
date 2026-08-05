#!/bin/bash
# #393 — start (or restart) one sync_loop per vast.ai box, and prove the
# first tick landed.
#
# Usage:  bash sync/launch_sync.sh [label ...]      # default: every box
#
# CLAUDE.md § Remote Machine Monitoring: every remote run has a sync loop
# for its whole duration, and the loop is verified by `ls` rather than by
# reading its own log — a missing failure line can just mean the pattern
# never matched anything.
#
# Each box gets its own local root. One shared root would have six loops
# writing the same results/ filenames (`run_cf393_<cell>.log` is per cell,
# but `ladder.csv` is not), and the last writer would win.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SL="$HERE/sync_loop.sh"

# label host port local-root-suffix
BOXES="a ssh2.vast.ai 11448 cf393_sync
b ssh4.vast.ai 13146 cf393_sync_b
c ssh6.vast.ai 18762 cf393_sync_c
d ssh7.vast.ai 18862 cf393_sync_d
e ssh5.vast.ai 18856 cf393_sync_e
f ssh1.vast.ai 18914 cf393_sync_f"

WANT="${*:-a b c d e f}"

while read -r lbl host port root; do
  [ -n "$lbl" ] || continue
  case " $WANT " in *" $lbl "*) ;; *) continue ;; esac

  # Match the local root with its slashes: box a's root `cf393_sync` is a
  # prefix of every other box's, so a bare substring test would report a
  # dead loop as alive the moment any other box's loop was running.
  if pgrep -af "sync_loop.sh" 2>/dev/null | grep -q "/$root/"; then
    echo "[$lbl] sync loop already running, leaving it"
    continue
  fi

  LOCAL="$HOME/$root/2026-08-04_ema_sched_ladder"
  mkdir -p "$LOCAL/sync" "$LOCAL/results"
  ( cd "$LOCAL" && \
    REMOTE_HOST="$host" REMOTE_PORT="$port" SSH_USER=root \
    REMOTE_DIR=/root/cf/experiments/2026-08-04_ema_sched_ladder \
    REMOTE_RUNS=/root/cf393_runs \
    LOCAL_DIR="$LOCAL" \
      nohup setsid bash "$SL" >> "$LOCAL/sync/sync_loop_vast${lbl}.log" 2>&1 < /dev/null & )
  echo "[$lbl] sync loop started -> $LOCAL"
done <<<"$BOXES"
