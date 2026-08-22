#!/bin/bash
# #404 — one progress line per tick for round 6, and every failure signature
# the moment it lands.
#
# It emits on stdout, one line per event, so an agent monitor turns each line
# into a notification. Silence is not success here: the filter carries the
# failure words as well as the progress, so a driver that aborts, a watchdog
# that fires and a checkpoint that does not read all reach the reader.
#
# Usage:  bash scripts/watch_round6.sh [seconds between ticks]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
R="$STUDY/results"
EVERY="${1:-1800}"
DRV="$R/round6.pid"
FIN_LOG="$R/finish_round6.log"
RND_LOG="$R/round6.log"
SIG='ABORT|WATCHDOG|TIMEOUT|^BAD |MISSING|SCORE |DONE|teardown|posted to PR|the push failed|pushed '

seen_r=0; seen_f=0
while :; do
  # Every new line of either log that a reader would act on.
  for pair in "$RND_LOG:r" "$FIN_LOG:f"; do
    log="${pair%:*}"; which="${pair##*:}"
    [ -f "$log" ] || continue
    n="$(grep -c '^' "$log")"
    if [ "$which" = r ]; then old=$seen_r; else old=$seen_f; fi
    if [ "$n" -gt "$old" ]; then
      tail -n +$(( old + 1 )) "$log" | grep -E "$SIG" | sed "s/^/[$which] /"
      if [ "$which" = r ]; then seen_r=$n; else seen_f=$n; fi
    fi
  done

  # One status line per tick, off the heartbeat's own probe.
  echo "[tick] $(ROUND=round6 bash "$HERE/heartbeat.sh" 2>/dev/null | cut -c1-400)"

  # The driver is the round. When it goes, the finisher takes over, and when
  # that goes too there is nothing left to watch.
  if [ -s "$DRV" ] && ! ps -p "$(cat "$DRV")" >/dev/null 2>&1; then
    if [ -f "$FIN_LOG" ] && grep -q 'FINISH DONE' "$FIN_LOG"; then
      echo "[end] the driver is gone and the finisher wrote FINISH DONE"
      exit 0
    fi
    echo "[note] the driver is gone — the finisher is still working"
  fi
  sleep "$EVERY"
done
