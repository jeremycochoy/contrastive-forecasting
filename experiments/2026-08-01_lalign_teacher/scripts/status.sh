#!/bin/bash
# #390 — one-screen status of the whole pipeline. Reads only disk state, so
# it is safe to run at any time and tells the truth even if a log lies.
#
#   WT=/home/jupyter/wt-cf-390-train bash status.sh          # full
#   ONELINE=1 bash status.sh                                 # heartbeat line
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

OUT="$WT/experiments/2026-08-01_lalign_teacher"
RUNS="$OUT/runs"; RES="$OUT/results"; EVALS="$OUT/eval_gm_mase"

pipe="down"
pid="$(cat "$RES/pipeline.pid" 2>/dev/null)"
[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pipe="up(pid $pid)"
# `pgrep -c` prints 0 AND exits 1 on no match, so an `|| echo 0` fallback
# appends a second line and every arithmetic use of the value breaks.
count_procs(){ pgrep -fc -- "$1" 2>/dev/null || true; }
trainers=$(count_procs "--run-name bb_small.*alignteacher"); trainers="${trainers:-0}"
heads=$(count_procs "--run-name qhead_2L"); heads="${heads:-0}"
evals=$(count_procs "eval_gift_eval_official.py"); evals="${evals:-0}"

if [ "${ONELINE:-0}" = 1 ]; then
  steps=""
  for arm in "${CF390_ARMS[@]}"; do
    name="$(bb_name "$arm")"
    s="$(grep -ohE '^\[ *[0-9]+\]' "$RES/run_${name}.log" 2>/dev/null | tr -dc '0-9\n' | tail -1)"
    steps="$steps ${arm}:${s:-0}"
  done
  done_cells=$(ls "$EVALS"/*_summary.txt 2>/dev/null | wc -l)
  nans=$(grep -l 'NaN/Inf DETECTED' "$RES"/run_*alignteacher.log 2>/dev/null | wc -l)
  echo "HEARTBEAT $(date '+%m-%d %H:%M') pipeline=$pipe bb=$trainers head=$heads eval=$evals cells_measured=$done_cells nan_runs=$nans steps:$steps"
  exit 0
fi

echo "=== #390 pipeline status $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "pipeline: $pipe   backbone trainers: $trainers   head trainers: $heads   gift-evals: $evals"
echo
printf '%-16s %10s %10s %10s  %s\n' arm step ck nan log
for arm in "${CF390_ARMS[@]}"; do
  name="$(bb_name "$arm")"
  tlog="$RES/run_${name}.log"
  s="$(grep -ohE '^\[ *[0-9]+\]' "$tlog" 2>/dev/null | tr -dc '0-9\n' | tail -1)"
  ck=$(ls -t "$RUNS/${name}"*_*k.pth 2>/dev/null | grep -v optimizer | head -1)
  nan="$(grep -ohE 'NaN/Inf DETECTED at step [0-9]+' "$tlog" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')"
  printf '%-16s %10s %10s %10s  %s\n' "$arm" "${s:-0}" \
    "$(basename "${ck:-none}" | grep -oE '_[0-9]+k' | tr -d '_' || echo none)" \
    "${nan:--}" "$(basename "$tlog")"
done
echo
echo "--- measured cells (97 configs each) ---"
for f in "$EVALS"/*_summary.txt; do
  [ -e "$f" ] || { echo "  none yet"; break; }
  # `head -1`: the summary's first line is the aggregate, the second
  # names the backbone it was measured on.
  printf '  %-34s %s\n' "$(basename "$f" _summary.txt)" "$(head -1 "$f")"
done
echo
echo "--- eval cells in flight ---"
for d in "$EVALS"/*/; do
  [ -d "$d" ] || { echo "  none"; break; }
  csv="$d/gift/all_results.csv"
  n=0; [ -f "$csv" ] && n=$(( $(wc -l < "$csv") - 1 ))
  printf '  %-34s %s/97 configs\n' "$(basename "$d")" "$n"
done
echo
tail -6 "$RES/pipeline.log" 2>/dev/null
