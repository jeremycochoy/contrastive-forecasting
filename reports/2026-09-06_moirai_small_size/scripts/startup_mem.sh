#!/bin/bash
# #412 — the GPU memory of one arm from its FIRST second, at full rate.
#
# WHY IT EXISTS. `results/trial/smoke.csv` reads 10,062 MiB for each k = 32
# arm. The live `k32_r200_08` holds 6,402, and 6,356 samples in 150 s find no
# transient above it. Two readings bound the question: session 9532bc caught
# that arm at 6,382 MiB between 12 and 72 seconds after it started, and the
# k = 3 smoke rows match their steady state within 56 MiB. So the unexplained
# window is the FIRST ~30 SECONDS of a k = 32 arm, before the trainer prints
# its first step line.
#
# WHICH ANSWER MATTERS. The smoke runs 150 steps, so its whole window is
# startup. If 10,062 is a real startup peak, the smoke row is the RIGHT number
# to gate on: a starting arm must fit its startup, and a gate sized on steady
# state lets an arm start and then die in its own warm-up. If instead the max
# never leaves steady state, the k = 32 smoke rows over-size every plan built
# on them by 3,660 MiB per arm.
#
# HOW IT CATCHES THE WINDOW. It polls for the pid every 0.2 s, not every 20 s,
# because a 20 s wait can miss the whole window and return a clean
# steady-state max that proves neither branch. It then samples `nvidia-smi` in
# a tight loop and writes every reading with its offset, so the curve is on
# disk and not just its maximum.
#
# Usage:  bash scripts/startup_mem.sh k8_r100_09 [seconds]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARM="${1:?usage: startup_mem.sh <arm> [seconds]}"
SECS="${2:-900}"
cf412_require_arm "$ARM" || exit 2
CSV="$CF412_RESULTS/startup_mem_${ARM}.csv"
LOG="$CF412_RESULTS/head_memory.log"
mkdir -p "$CF412_RESULTS"

# The trainer of THIS arm. `--run-name` is unique per arm, and no watcher
# shell carries it, so this cannot match a peer session's watcher.
K="$(cf412_depth "$ARM")"
NAME="$(printf 'cf393_%s_cf373k%s_cf412_%s' "$CF412_CELL" "$K" "$ARM")"
find_pid(){ ps -eo pid,args --no-headers \
  | awk -v n="$NAME" '$2 ~ /python/ && $0 ~ ("--run-name " n "( |$)") {print $1; exit}'; }

pid=""
while [ -z "$pid" ]; do pid="$(find_pid)"; [ -n "$pid" ] || sleep 0.2; done
t0="$(date +%s)"
echo "offset_s,used_mib" >"$CSV"
first=""; max=0; n=0
while kill -0 "$pid" 2>/dev/null; do
  now="$(date +%s)"
  [ $(( now - t0 )) -ge "$SECS" ] && break
  m="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        2>/dev/null | awk -F', *' -v p="$pid" '$1==p{print $2}')"
  if [ -n "$m" ]; then
    printf '%s,%s\n' "$(( now - t0 ))" "$m" >>"$CSV"
    [ -z "$first" ] && first="$m"
    [ "$m" -gt "$max" ] && max="$m"
    n=$(( n + 1 ))
  fi
done
smoke="$(awk -F',' -v a="$ARM" '$1==a {print $7}' "$CF412_RESULTS/trial/smoke.csv" 2>/dev/null)"
printf '%s %s startup: first %s MiB, MAX %s MiB over %s samples, smoke row %s\n' \
  "$(date '+%m-%d %H:%M:%S')" "$ARM" "${first:-?}" "$max" "$n" "${smoke:-?}" | tee -a "$LOG"
