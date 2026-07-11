#!/bin/bash
# #369 follow-up leg — wait for a fully free GPU, then extend the arm-C
# B=1024 backbone 37,500 -> 50,000 and arm the head chain on the other
# GPU (2L/6L at steps 40000, 45000, 50000).
#
# "Free" = < 1000 MiB used for 3 consecutive 60 s checks (B=1024 BB
# needs ~20.4 GiB; it cannot co-reside with any #371 cell).
set -uo pipefail
: "${WT:?}"; : "${OUT:?}"
LOG="$OUT/results/extend_queue.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [queue-50k] $*" >>"$LOG"; }

if pgrep -f 'extend_bb_to_25k.sh' >/dev/null 2>&1; then
  log "another extend is already running — exiting"; exit 0
fi

log "waiting for a free GPU (need 3 consecutive <1000MiB checks)"
declare -A streak=( [0]=0 [1]=0 )
GPU=""
while :; do
  for g in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
    if [ "$used" -lt 1000 ]; then
      streak[$g]=$(( ${streak[$g]} + 1 ))
    else
      streak[$g]=0
    fi
    if [ "${streak[$g]}" -ge 3 ]; then GPU="$g"; break 2; fi
  done
  sleep 60
done
log "GPU $GPU free — launching BB extend to 50000"

setsid nohup env WT="$WT" OUT="$OUT" GPU="$GPU" STEPS=50000 \
  bash "$OUT/scripts/extend_bb_to_25k.sh" \
  >>"$OUT/results/extend_bb_orchestrator.log" 2>&1 </dev/null &
BB_PID=$!
log "BB extend pid=$BB_PID on GPU $GPU"

OTHER=$(( 1 - GPU ))
setsid nohup env WT="$WT" OUT="$OUT" \
  bash "$OUT/scripts/dl_chain.sh" "$OTHER" \
  2L@40000 6L@40000 2L@45000 6L@45000 2L@50000 6L@50000 \
  >>"$OUT/results/dl_chain_50k.log" 2>&1 </dev/null &
log "head chain armed on GPU $OTHER (waits on each _step<N>.pth)"
