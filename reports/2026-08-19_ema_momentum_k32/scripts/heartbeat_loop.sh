#!/bin/bash
# #404 — the hourly backstop behind round 3c.
#
# `heartbeat.sh` prints one liveness line and exits. This loop calls it every
# hour and appends the line to `results/heartbeat.log`, so a reader who arrives
# at any hour sees whether the counter moved between two probes.
#
# WHY A LOOP AND NOT A NOTIFICATION. A hung process and a slow one look the
# same to a watcher that waits on an exit code. On 2026-06-11 a box spent a
# night "running" at 0% and the money drained. The probe reads the counter, not
# the process table.
#
# WHY IT IS DETACHED. A probe that lives inside an agent session dies with the
# session, and the run it watches does not. Launch it with:
#
#   nohup setsid bash scripts/heartbeat_loop.sh > /dev/null 2>&1 < /dev/null &
#
# It stops on its own when the round 3c driver leaves the process table.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
R="$STUDY/results"
EVERY="${EVERY:-3600}"
LOG="$R/heartbeat.log"

echo "$$" > "$R/heartbeat.pid"
echo "[$(date '+%m-%d %H:%M')] #404 heartbeat loop up, every ${EVERY}s" >> "$LOG"

while :; do
  bash "$HERE/heartbeat.sh" >> "$LOG" 2>&1
  # The driver owns the round. When it exits, the last probe is written above
  # and this loop has nothing left to watch.
  if [ -s "$R/round3c.pid" ] && ! ps -p "$(cat "$R/round3c.pid")" >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] #404 the round 3c driver is gone — heartbeat loop stops" >> "$LOG"
    exit 0
  fi
  sleep "$EVERY"
done
