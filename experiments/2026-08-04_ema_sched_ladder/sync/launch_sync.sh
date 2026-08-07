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
f ssh1.vast.ai 18914 cf393_sync_f
g ssh7.vast.ai 13258 cf393_sync_g"

WANT="${*:-a b c d e f g}"

while read -r lbl host port root; do
  [ -n "$lbl" ] || continue
  case " $WANT " in *" $lbl "*) ;; *) continue ;; esac

  # Identify a running loop by its working directory, not its argv. Every
  # loop runs the same `bash .../sync_loop.sh` command line and takes its
  # target from the environment, so argv cannot tell two of them apart —
  # only the loops started via a wrapper shell happen to carry the path,
  # and matching on that reports the others as dead and starts a second
  # copy against the same local root. /proc/PID/cwd is what actually differs.
  running=""
  for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$HOME/$root/2026-08-04_ema_sched_ladder" ] \
      && { running="$p"; break; }
  done
  if [ -n "$running" ]; then
    echo "[$lbl] sync loop already running (pid $running), leaving it"
    continue
  fi

  LOCAL="$HOME/$root/2026-08-04_ema_sched_ladder"
  mkdir -p "$LOCAL/sync" "$LOCAL/results"

  # sync_loop.sh resolves safe_pull.sh as a sibling of LOCAL_DIR, and exits
  # at once if it is not there. Every local root therefore needs its own
  # copy — the loop aborts on the first tick otherwise, which looks exactly
  # like a loop that is running and finding nothing.
  SP="$(dirname "$LOCAL")/2026-04-27_periodic-synth-mix/scripts"
  mkdir -p "$SP"
  cp -f "$HERE/../../2026-04-27_periodic-synth-mix/scripts/safe_pull.sh" "$SP/" \
    || { echo "[$lbl] ABORT: cannot stage safe_pull.sh"; continue; }
  ( cd "$LOCAL" && \
    REMOTE_HOST="$host" REMOTE_PORT="$port" SSH_USER=root \
    REMOTE_DIR=/root/cf/experiments/2026-08-04_ema_sched_ladder \
    REMOTE_RUNS=/root/cf393_runs \
    LOCAL_DIR="$LOCAL" \
      nohup setsid bash "$SL" >> "$LOCAL/sync/sync_loop_vast${lbl}.log" 2>&1 < /dev/null & )
  echo "[$lbl] sync loop started -> $LOCAL"
done <<<"$BOXES"
