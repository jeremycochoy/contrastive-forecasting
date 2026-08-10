#!/bin/bash
# #373 round 2 — ask every box what it is actually running.
#
# Usage: bash r2_verify.sh
#
# Four things, per box, all read off the box rather than off a launcher log:
#
#   1. how many `train.py` processes. More than one means two runs are
#      writing the same run name into the same directory, and round 1 lost
#      45 minutes of a 5090 to exactly that.
#   2. `--train-rollout-depth`. A cell silently running k = 0 would produce a
#      full set of plausible numbers that answer nothing.
#   3. the run name and the step, so a stalled run is visible.
#   4. how many heads are training beside it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$(dirname "$HERE")/results"
BOXES="$RES/r2_boxes.tsv"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)

printf '%-4s %-6s %-4s %-6s %-9s %s\n' CELL TRAIN k HEADS STEP RUN
bad=0
while IFS=$'\t' read -r cell id host port label stops; do
  [ -n "${cell:-}" ] || continue
  case "$cell" in \#*) continue;; esac
  # Everything through `ps | grep -c`, and the bracket in `[f]req` keeps the
  # pattern from matching the shell that carries it: the counters read 3
  # instead of 1 until it was there, because the ssh command line holds the
  # string it greps for. `pgrep -c` also prints 0 AND exits 1 on no match,
  # which turned a guard into a second line of output; ps does not.
  out=$(timeout 40 ssh "${SSH_OPTS[@]}" -n -p "$port" "root@$host" '
    all=$(ps -eo args 2>/dev/null)
    line=$(printf "%s\n" "$all" | grep -- "[f]req-embedding/scripts/train.py" | head -1)
    n=$(printf "%s\n" "$all" | grep -c -- "[f]req-embedding/scripts/train.py")
    h=$(printf "%s\n" "$all" | grep -c -- "[t]rain_forecasting_head.py")
    k=$(printf "%s" "$line" | grep -o -- "--train-rollout-depth [0-9]*" | head -1 | cut -d" " -f2)
    r=$(printf "%s" "$line" | grep -o -- "--run-name [^ ]*" | head -1 | cut -d" " -f2)
    s=$(find /root/cf373_runs -name "*losses*.csv" -exec tail -1 {} \; 2>/dev/null \
        | cut -d, -f1 | grep -E "^[0-9]+$" | sort -n | tail -1)
    printf "%s|%s|%s|%s|%s\n" "${n:-0}" "${k:--}" "${h:-0}" "${s:--}" "${r:--}"' 2>/dev/null | tail -1)
  IFS='|' read -r n k h s r <<<"${out:-?|?|?|?|unreachable}"
  printf '%-4s %-6s %-4s %-6s %-9s %s\n' "$cell" "$n" "$k" "$h" "$s" "${r:0:52}"
  [ "$n" = "1" ] || { echo "    ^ WARNING: $n train.py on $cell's box"; bad=1; }
  [ "$k" = "3" ] || { echo "    ^ WARNING: depth reads '$k', want 3"; bad=1; }
done < "$BOXES"
[ "$bad" -eq 0 ] && echo "all boxes: exactly one backbone, at k = 3"
exit "$bad"
