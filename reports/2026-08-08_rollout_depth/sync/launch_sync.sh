#!/bin/bash
# #373 — one sync_loop per vast.ai box, and proof its first tick landed.
#
# Usage:  bash sync/launch_sync.sh [label ...]      # default: every box
#
# CLAUDE.md § Remote Machine Monitoring: every remote run has a sync loop
# for its whole duration, and the loop is verified by `ls` rather than by
# reading its own log — a missing failure line can just mean the pattern
# never matched anything. `verify_sync.sh` is that `ls`.
#
# Each box gets its own local root. One shared root would have three loops
# writing the same `results/queue.log` name, and the last writer would win.
#
# Boxes and their SSH addresses come from results/boxes.tsv, written by
# launch_box.sh, so this file never carries a hard-coded port that a
# re-provisioned box has already invalidated.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
SL="$HERE/sync_loop.sh"
BOXES="${BOXES_FILE:-$STUDY/results/boxes.tsv}"
LOCAL_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
SAFE_PULL="${SAFE_PULL:-$STUDY/../../experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh}"

[ -f "$BOXES" ] || { echo "ABORT: no box table at $BOXES" >&2; exit 2; }
[ -f "$SAFE_PULL" ] || { echo "ABORT: no safe_pull.sh at $SAFE_PULL" >&2; exit 2; }
WANT="${*:-}"

while IFS=$'\t' read -r lbl id host port jobs; do
  case "$lbl" in ''|'#'*) continue ;; esac
  if [ -n "$WANT" ]; then
    case " $WANT " in *" $lbl "*) ;; *) continue ;; esac
  fi

  LOCAL="$LOCAL_BASE/$lbl"

  # Identify a running loop by its working directory, not its argv: every
  # loop runs the same `bash .../sync_loop.sh` and takes its target from the
  # environment, so argv cannot tell two of them apart and matching on it
  # starts a second copy against the same local root.
  running=""
  for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$LOCAL" ] && { running="$p"; break; }
  done
  if [ -n "$running" ]; then
    echo "[$lbl] sync loop already running (pid $running), leaving it"
    continue
  fi

  mkdir -p "$LOCAL/sync" "$LOCAL/results"
  ( cd "$LOCAL" && \
    REMOTE_HOST="$host" REMOTE_PORT="$port" SSH_USER=root \
    REMOTE_DIR=/root/cf/reports/2026-08-08_rollout_depth \
    REMOTE_RUNS=/root/cf373_runs \
    LOCAL_DIR="$LOCAL" SAFE_PULL="$SAFE_PULL" \
      nohup setsid bash "$SL" >> "$LOCAL/sync/sync_loop_${lbl}.log" 2>&1 < /dev/null & )
  echo "[$lbl] sync loop started -> $LOCAL (instance $id, $host:$port)"
done < "$BOXES"
