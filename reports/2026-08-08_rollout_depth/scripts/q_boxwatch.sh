#!/bin/bash
# #373 round 3 — return when the box stops answering, or at the deadline.
#
# The dispatcher polls for a `.rc` file on the box. A box that dies never
# writes one, so the queue would wait on it forever. This is the thing that
# notices. Usage: bash q_boxwatch.sh <host> <port> [max seconds]
set -uo pipefail
H="${1:?host}"; P="${2:?port}"; MAX="${3:-86400}"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results"
miss=0; t=0
while [ "$t" -lt "$MAX" ]; do
  sleep 300; t=$(( t + 300 ))
  if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
         -o ConnectTimeout=25 -o BatchMode=yes -p "$P" "root@$H" true 2>/dev/null; then
    miss=0
  else
    miss=$(( miss + 1 ))
    echo "[$(date -u '+%H:%M:%SZ')] box unreachable ($miss)" | tee -a "$RES/q_boxwatch.log"
    [ "$miss" -ge 3 ] && { echo "BOX DOWN after $miss checks"; exit 3; }
  fi
done
echo "box alive at deadline"
