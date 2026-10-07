#!/bin/bash
# #425 — the probe of one full wave on the box: the first wave of each lane
# of queue.sh, as the queue plans it, for a few hundred steps and with no
# score. It gives the step rate and the GPU memory of a wave, for the wave
# sizes and the time estimate of the queue. The heads go to a test folder
# that the queue never reads.
#
# For each wave: the last reports of its trainer (the step rate, the share of
# time that it waited for its stream, the memory that torch holds), and the
# peak memory of its process on the GPU. Other processes can use the GPU
# during a probe, so the probe also gives the peak of the whole GPU.
#
# Usage, on the box:  bash probe.sh [steps]      (default 500)
#   HEAD_LOG_EVERY=50 CF425_HEAD_ARCH=linear CF425_WAVE_SIZE=47 bash probe.sh 200
#                       a wave of 47 linear heads and the old-data wave, with
#                       a report every 50 steps
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${1:-500}"
JOBS="${CF425_JOBS:-$HERE/jobs.tsv}"
case "${CF425_HEAD_ARCH:-transformer}" in linear) NAME=cf-425-lin ;; *) NAME=cf-425 ;; esac
RES="${CF425_PROBE_RES:-/workspace/results/$NAME/probe}"
ROOT="${CF425_PROBE_ROOT:-/workspace/ckpt/$NAME/probe}"

mkdir -p "$RES"
# The jobs of wave 1 of each lane, from the plan of the queue: "<code> <stop>k".
CF425_DRY_RUN=1 bash "$HERE/queue.sh" | awk '$2 == 1 { print $4, $5 }' >"$RES/wave1.txt"
awk -F'\t' 'NR == FNR { want[$1] = 1; next }
  /^#/ || ($1 " " $3 "k") in want' "$RES/wave1.txt" "$JOBS" >"$RES/jobs.tsv"
echo "probe: $(grep -vc '^#' "$RES/jobs.tsv") jobs, $STEPS steps"

# Every 10 s: the memory and the use of the whole GPU, and the memory of the
# process of each wave of the probe.
( while :; do
    now=$(date +%T)
    echo "$now total $(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)"
    for pid in $(pgrep -f "$RES/queue/waves/"); do
      wave=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null \
        | grep -o 'waves/[^/ ]*' | head -1)
      mib=$(nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits | awk -F', *' -v p="$pid" '$1 == p { print $2 }')
      [ -n "$wave" ] && [ -n "$mib" ] && echo "$now wave ${wave#waves/} $mib"
    done
    sleep "${CF425_PROBE_SAMPLE:-10}"
  done ) >"$RES/gpu.log" &
sampler=$!
CF425_JOBS="$RES/jobs.tsv" CF425_HEAD_STEPS="$STEPS" CF425_SCORE=0 \
  CF425_ROOT="$ROOT" CF425_RES="$RES/queue" \
  bash "$HERE/queue.sh" >"$RES/queue.out" 2>&1
echo "queue rc=$?"
kill "$sampler"

for log in "$RES"/queue/waves/*/train.log; do
  wave=$(basename "$(dirname "$log")")
  echo "$wave: $(grep -c . "$(dirname "$log")/jobs.jsonl") jobs"
  grep '^\[shared\]' "$log" | tail -2
  awk -v w="$wave" '$2 == "wave" && $3 == w && $4 > m { m = $4 }
    END { print "peak " m + 0 " MiB of GPU memory for the process of this wave" }' "$RES/gpu.log"
done
awk '$2 == "total" { gsub(",", "", $3); if ($3 > m) m = $3 }
  END { print "peak GPU memory, all processes:", m + 0, "MiB" }' "$RES/gpu.log"
