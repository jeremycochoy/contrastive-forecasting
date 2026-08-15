#!/bin/bash
# #401 — the sync loop for a leg that trains on a rented box.
#
# CLAUDE.md § Remote Machine Monitoring: EVERY remote training run has a sync
# loop for its whole duration, short runs included, and the loop is verified
# by `ls` rather than by reading its own log.
#
# The loop itself is #373's `sync/sync_loop.sh`, unchanged. It is already
# parameterised by environment — remote host, remote study dir, remote runs
# root, local dir — and it carries the per-class size floors this study's
# artefacts need, measured against this exact backbone and this exact head
# arch. A copy here would be a second set of floors to keep in step.
#
# REMOTE_RUNS is this study's root, and this study saves ONE LEVEL DEEPER
# than #373: `$CF401_ROOT/k<K>/<cell>/leg_<N>k/`, because `cf401_arm_root`
# adds the depth. A loop that walked a fixed depth would pull nothing, and
# the first-tick check below would say so only after 10 minutes. It does not:
# `remote_listing` runs `find <dir> -type f` with no `-maxdepth`, and
# `pull_tree` rebuilds the relative path, so any depth comes back. Verified
# against a tree with this study's exact layout — run
# `bash sync/verify_glob_depth.sh` to repeat it.
#
# When this is NOT needed: a leg that trains on elisa writes straight to
# /home/jupyter/checkpoints_backup/cf-401, which is durable local disk on the
# same machine. There is nothing to pull. Run this only for a rented box.
#
# Usage:
#   REMOTE_HOST=ssh5.vast.ai REMOTE_PORT=12345 SSH_USER=root \
#   REMOTE_DIR=/root/contrastive-forecasting/reports/2026-08-15_rollout_depth_k16_8_32 \
#     bash sync/launch_sync.sh box_a
#
# The label names the local root, so two boxes never write one another's
# files. The script waits for the first tick and then lists what landed.
set -uo pipefail

LABEL="${1:?usage: launch_sync.sh <box label>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$(dirname "$HERE")/scripts/study.sh"

LOOP="$CF401_PARENT/sync/sync_loop.sh"
SAFE_PULL="${SAFE_PULL:-$CF401_REPO/experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/cf401_sync/$LABEL}"
INTERVAL="${INTERVAL:-900}"

[ -f "$LOOP" ] || { echo "ABORT: no sync loop at $LOOP" >&2; exit 2; }
[ -f "$SAFE_PULL" ] || { echo "ABORT: no safe_pull.sh at $SAFE_PULL" >&2; exit 2; }
: "${REMOTE_HOST:?REMOTE_HOST must be set}"
: "${REMOTE_DIR:?REMOTE_DIR must be set: this study directory on the box}"

case "$LOCAL_DIR" in
  /tmp|/tmp/*) echo "ABORT: LOCAL_DIR=$LOCAL_DIR is ephemeral" >&2; exit 2 ;;
esac

# One loop per local root, identified by its working directory. Every loop
# runs the same command line and takes its target from the environment, so
# argv cannot tell two of them apart.
for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
  if [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$LOCAL_DIR" ]; then
    echo "[$LABEL] a loop already runs here (pid $p), leaving it"
    exit 0
  fi
done

mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR" || exit 2
REMOTE_HOST="$REMOTE_HOST" REMOTE_PORT="${REMOTE_PORT:-22}" \
REMOTE_DIR="$REMOTE_DIR" REMOTE_RUNS="${REMOTE_RUNS:-$CF401_ROOT}" \
LOCAL_DIR="$LOCAL_DIR" SSH_USER="${SSH_USER:-root}" \
SAFE_PULL="$SAFE_PULL" INTERVAL="$INTERVAL" \
  nohup setsid bash "$LOOP" >"$LOCAL_DIR/sync_loop.log" 2>&1 &
# setsid can fork, so `$!` is not reliably the loop. Report the process whose
# working directory is this local root — the same test the duplicate check
# above uses.
sleep 2
pid="$(for p in $(pgrep -f "bash .*sync_loop.sh" 2>/dev/null); do
         [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$LOCAL_DIR" ] && echo "$p"
       done | head -1)"
echo "[$LABEL] sync loop pid ${pid:-?} -> $LOCAL_DIR"

# Verify by `ls`, not by reading the loop's log: a missing failure line can
# mean the pattern never matched. A first tick is 2 to 5 minutes.
echo "[$LABEL] waiting for the first tick"
for _ in $(seq 1 60); do
  sleep 10
  if [ -n "$(find "$LOCAL_DIR" -type f ! -name sync_loop.log -print -quit 2>/dev/null)" ]; then
    echo "[$LABEL] first tick landed:"
    find "$LOCAL_DIR" -type f ! -name sync_loop.log -printf '  %s\t%p\n' | head -40
    exit 0
  fi
done
echo "[$LABEL] WARNING: nothing landed in 10 minutes — check $LOCAL_DIR/sync_loop.log" >&2
exit 1
