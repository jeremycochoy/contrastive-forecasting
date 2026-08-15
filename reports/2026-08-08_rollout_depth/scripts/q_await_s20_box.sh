#!/bin/bash
# #373 round 3 — block until the rented box is gone, then report the meter.
# q_finish.sh owns the destroy; this only watches for it and prints.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")/../results" && pwd)"
BOX_ID="${BOX_ID:-47557391}"
DEADLINE=$(( SECONDS + 5400 ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if ! timeout 90 vastrun-status 2>/dev/null | grep -q "$BOX_ID"; then
    echo "=== BOX $BOX_ID GONE $(date -u +%H:%M:%SZ) ==="
    timeout 90 vastrun-balance 2>&1 | head -3
    tail -6 "$RES/q_finish.log"
    exit 0
  fi
  sleep 120
done
echo "=== DEADLINE: box $BOX_ID still up at $(date -u +%H:%M:%SZ) ==="
timeout 90 vastrun-status 2>&1 | head -5
tail -8 "$RES/q_finish.log"
