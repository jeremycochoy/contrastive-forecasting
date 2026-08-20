#!/bin/bash
# #404 — the hourly backstop behind the round that runs now.
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
# It stops on its own when the round's driver leaves the process table.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
R="$STUDY/results"
ROUND="${ROUND:-round4}"
PIDF="${ROUND_PID:-$R/$ROUND.pid}"
EVERY="${EVERY:-3600}"
LOG="$R/heartbeat.log"

echo "$$" > "$R/heartbeat.pid"
echo "[$(date '+%m-%d %H:%M')] #404 heartbeat loop up for $ROUND, every ${EVERY}s" >> "$LOG"

while :; do
  ROUND="$ROUND" ROUND_PID="$PIDF" bash "$HERE/heartbeat.sh" >> "$LOG" 2>&1
  # The driver owns the round. When it exits, the last probe is written above
  # and this loop has nothing left to watch.
  if [ -s "$PIDF" ] && ! ps -p "$(cat "$PIDF")" >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M')] #404 the $ROUND driver is gone — heartbeat loop stops" >> "$LOG"
    exit 0
  fi
  sleep "$EVERY"
done
