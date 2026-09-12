#!/bin/bash
# #412 — the GPU memory ONE head holds, sampled for its whole train.
#
# WHY. `CF412_HEAD_VRAM_MIB` is 9,000, and no row of this card says where that
# number comes from. A head that cannot get 9,000 MiB waits 4 hours and aborts
# with no score, and on 2026-09-10 that cost arm `k3_r100_09_lr56_dec` its
# 40,000-step head on a card holding 7,018 MiB free.
#
# So this samples what a head of THIS shape actually holds, over its whole
# train, and writes the peak. A gate can then rest on a measurement.
#
# CAUTION. A steady reading is not a settled one. This card has a k = 32 leg
# that ran 3,660 MiB above a reading taken hours into the same leg. Sample the
# WHOLE train, and read the peak, never the last row.
#
# Usage:  bash scripts/head_vram_sample.sh <arm> <stop> [poll seconds]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
. "$HERE/pass4_lib.sh"

ARM="${1:?usage: head_vram_sample.sh <arm> <stop> [poll]}"
STOP="${2:?usage: head_vram_sample.sh <arm> <stop> [poll]}"
POLL="${3:-60}"
OUT="$CF412_RESULTS/head_vram_${ARM}_bb$(( STOP / 1000 ))k.csv"
mkdir -p "$CF412_RESULTS"
echo "time,pid,used_mib,gpu,gpu_free_mib" >"$OUT"

peak=0
while :; do
  pid="$(cf412_head_pid "$ARM" "$STOP")" || break
  gpu="$(cf412_head_gpu "$pid")"
  used=""
  for p in $(cf412_descendants "$pid"); do
    m="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
          2>/dev/null | awk -F, -v q="$p" '$1+0==q{gsub(/[^0-9]/,"",$2); print $2}')"
    [ -n "$m" ] && used="$m"
  done
  if [ -n "$used" ]; then
    [ "$used" -gt "$peak" ] && peak="$used"
    echo "$(date '+%H:%M:%S'),$pid,$used,$gpu,$(cf412_gpu_free "$gpu")" >>"$OUT"
  fi
  sleep "$POLL"
done
echo "# peak_mib,$peak" >>"$OUT"
echo "PEAK $peak MiB — $OUT"
