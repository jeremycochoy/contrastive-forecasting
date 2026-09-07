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
# WHAT THE MEASUREMENT FOUND, and what it did not. `k32_r200_08` held 6,402
# MiB for four hours and then took 10,418, ABOVE its 10,062 smoke row. So the
# smoke row is not an over-statement and the gate stays. The CAUSE is not
# known. A commit subject of this card once named the step-20,000 save as the
# mechanism, and that was too strong: the sampling window was 30 minutes, or
# 2,160 steps, and step 20,000 is only one of them. Two k = 3 arms cross the
# same save with no growth, and the latent-drift probe at that step runs the
# encoder path under no-grad with its cache on CPU, so neither explains a
# depth-specific jump.
#
# THE LESSON IS THE STEADY READING. Three samplers agreed on 6,402: a 60 s
# poll, a 30 s poll, and a tight loop of 6,356 samples in 150 seconds. All
# three were right and none measured the thing that mattered, because the arm
# changed four hours later. A steady reading is not a settled one.
#
# WHICH ANSWER MATTERS. The smoke runs 150 steps, so its whole window is
# startup. If 10,062 is a real startup peak, the smoke row is the RIGHT number
# to gate on: a starting arm must fit its startup, and a gate sized on steady
# state lets an arm start and then die in its own warm-up. If instead the max
# never leaves steady state, the k = 32 smoke rows over-size every plan built
# on them by 3,660 MiB per arm.
#
# HOW TO READ THE CURVE ON k8_r100_09. Three outcomes, not two. A k = 8 arm
# holds more than a k = 3 arm for real reasons, so a flat curve is only
# evidence of over-statement if it sits near the k = 3 figure. Read it against
# BOTH 7,160, its own smoke row, and 6,472, the k = 3 steady state.
#
#   1. A spike above 7,160, then a fall. The startup peak is real, and every
#      smoke row is the right number to gate on. Keep every gate.
#   2. Flat near 7,160. The k = 8 smoke row is CORRECT, there is no
#      over-statement at k = 8, and the k = 32 gap stays unexplained. Neither
#      branch wins.
#   3. Flat near 6,500, well under 7,160. The smoke over-states with depth, so
#      the k = 32 row is wrong the same way.
#
# Outcome 2 is the likeliest and it settles nothing. Do not read it as 3.
#
# THE CLEAN EXPERIMENT THIS CARD WILL NOT RUN. The gap lives at k = 32, so the
# measurement belongs on a k = 32 arm from its first second. No k = 32 arm is
# due to start again here, and a 9.6-hour leg is far too much to spend on a
# memory gate. This card infers from k = 8 and says so.
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

# FINE FOR THE FIRST `FINE_S` SECONDS, THEN COARSE TO THE END OF THE LEG.
# The first version sampled only the first 900 s, and that cannot see what
# this arm class actually does: `k32_r200_08` held 6,402 MiB for four hours
# and then took 10,418 at its step-20,000 save, ABOVE its 10,062 smoke row.
# A tight loop spawns `nvidia-smi` about 42 times a second, which is fine for
# two minutes and not for six hours, so the rate drops after the startup
# window.
FINE_S="${CF412_STARTUP_FINE_S:-120}"
COARSE_S="${CF412_STARTUP_COARSE_S:-30}"

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
    if [ "$m" -gt "$max" ]; then
      max="$m"
      printf '%s %s new max %s MiB at %ss\n' "$(date '+%m-%d %H:%M:%S')" \
        "$ARM" "$m" "$(( now - t0 ))" >>"$LOG"
    fi
    n=$(( n + 1 ))
  fi
  [ $(( now - t0 )) -ge "$FINE_S" ] && sleep "$COARSE_S"
done
smoke="$(awk -F',' -v a="$ARM" '$1==a {print $7}' "$CF412_RESULTS/trial/smoke.csv" 2>/dev/null)"
printf '%s %s startup: first %s MiB, MAX %s MiB over %s samples, smoke row %s\n' \
  "$(date '+%m-%d %H:%M:%S')" "$ARM" "${first:-?}" "$max" "$n" "${smoke:-?}" | tee -a "$LOG"
