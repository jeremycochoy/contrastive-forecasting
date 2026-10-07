#!/bin/bash
# #425 — the probe of one full wave on the box: the first wave of each lane
# of queue.sh, as the queue plans it, for a few hundred steps and with no
# score. It gives the step rate and the GPU memory of a wave, for the wave
# sizes and the time estimate of the queue. The heads go to a test folder
# that the queue never reads.
#
# Usage, on the box:  bash probe.sh [steps]      (default 500)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${1:-500}"
RES="${CF425_PROBE_RES:-/workspace/results/cf-425/probe}"
ROOT="${CF425_PROBE_ROOT:-/workspace/ckpt/cf-425/probe}"

mkdir -p "$RES"
# The jobs of wave 1 of each lane, from the plan of the queue.
CF425_DRY_RUN=1 bash "$HERE/queue.sh" | awk '$2 == 1 { print $6 }' >"$RES/tags.txt"
awk -F'\t' 'NR == FNR { want[$1] = 1; next }
  /^#/ || ($2 "_bb" $3 "k_h30k_recon") in want' "$RES/tags.txt" \
  "$HERE/jobs.tsv" >"$RES/jobs.tsv"
echo "probe: $(grep -vc '^#' "$RES/jobs.tsv") jobs, $STEPS steps"

# The GPU memory and use, every 10 s.
( while :; do
    echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits)"
    sleep 10
  done ) >"$RES/gpu.log" &
sampler=$!
CF425_JOBS="$RES/jobs.tsv" CF425_HEAD_STEPS="$STEPS" CF425_SCORE=0 \
  CF425_ROOT="$ROOT" CF425_RES="$RES/queue" \
  bash "$HERE/queue.sh" >"$RES/queue.out" 2>&1
echo "queue rc=$?"
kill "$sampler"

for log in "$RES"/queue/waves/*/train.log; do
  echo "$(basename "$(dirname "$log")"): $(grep -c '^\[.*\] Backbone loaded' "$log") jobs"
  grep '^\[shared\]' "$log" | tail -2
done
awk '{ gsub(",", "", $2); if ($2 > m) m = $2 } END { print "peak GPU memory:", m, "MiB" }' \
  "$RES/gpu.log"
