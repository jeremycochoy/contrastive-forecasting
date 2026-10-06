#!/bin/bash
# #414 — extend `k3_r100_09_lr56_dec` past 200,000 steps, on the lane that
# `fix099_dec10k` frees.
#
# That arm improves from 100,000 to 200,000 steps, 1.3507 to 1.2979. The move
# is 0.0528 and the seed band is 0.0649, so the improvement is not yet ranked.
# Only more steps can rank it.
#
# It waits on ONE pid and then starts. It holds no poll loop over a pattern.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_PID="${1:?pid to wait on}"
# The lane is chosen when the wait ends, not when the waiter starts. Both
# cards carry other work, and the one that is free now need not be free then.
pick_gpu(){
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' '{ free = $3 - $2; if (free > best) { best = free; pick = $1 } }
                   END { print pick, best }'
}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
# The head the driver of the finished arm starts next claims memory too. Wait
# for a card that holds the 12000 MiB gate of a leg before reading the winner.
while true; do
  read -r GPU FREE <<<"$(pick_gpu)"
  [ "${FREE:-0}" -ge 12000 ] && break
  sleep 120
done
echo "[$(date '+%m-%d %H:%M:%S')] pid $WAIT_PID gone — gpu $GPU holds $FREE MiB"
cd "$HERE/.."
exec env ARMS=k3_r100_09_lr56_dec \
  CF412_STOPS="40000 100000 200000 300000 400000" \
  STOPS="300000 400000" \
  BB_GPU="$GPU" CF412_SAVE_EVERY=30000 \
  bash scripts/phase1.sh
