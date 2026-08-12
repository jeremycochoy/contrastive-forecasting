#!/bin/bash
# #373 round 2 — one line per tick: what is running, what it costs, and
# whether any box is billing without working.
#
# Usage: bash r2_heartbeat.sh [interval seconds]
#
# Written for a monitor, so it prints ONE line per tick on stdout and keeps
# going. A box that is up and holds no train.py and no head is the failure
# this exists to catch: cf373r2-b8 billed 37.6 hours at 0% utilisation in
# round 1 because nothing ever bootstrapped it. The line names such a box
# by id, so the next tick is not the first anyone hears of it.
set -uo pipefail

INTERVAL="${1:-1800}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
ROOT="$(cd "$HERE" && git rev-parse --show-toplevel)"
# A box gets this long from provisioning to its first work before the line
# calls it idle. The kit's image pull alone runs several minutes.
IDLE_GRACE_MIN="${IDLE_GRACE_MIN:-15}"

while :; do
  st="$(cd "$ROOT" && timeout 120 vastrun-status 2>/dev/null)"
  cred="$(cd "$ROOT" && timeout 60 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')"
  n_box=$(grep -c 'cf373r2-' <<<"$st")
  rate=$(awk '/cf373r2-/{gsub(/\$/,"",$6); s+=$6} END {printf "%.2f", s+0}' <<<"$st")

  done_n=$(python3 "$HERE/r2_coverage.py" 2>/dev/null | awk '/^deliverables/{print $4}')
  miss_n=$(python3 "$HERE/r2_coverage.py" 2>/dev/null | awk '/^deliverables/{print $6}')

  # Boxes past the grace period that hold neither a backbone nor a head.
  idle=""
  while read -r id label up; do
    [ -n "${id:-}" ] || continue
    mins=$(sed -E 's/h.*//;s/m$//' <<<"$up")
    case "$up" in *h*) mins=$(( ${mins:-0} * 60 + 60 ));; esac
    [ "${mins:-0}" -ge "$IDLE_GRACE_MIN" ] || continue
    row="$(grep -P "\t$id\t" "$RES/r2_boxes.tsv" 2>/dev/null | head -1)"
    if [ -z "$row" ]; then idle="$idle $label($id,no-row)"; continue; fi
    host=$(cut -f3 <<<"$row"); port=$(cut -f4 <<<"$row")
    busy=$(timeout 60 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
             -o ConnectTimeout=10 -n -p "$port" "root@$host" \
             'ps -eo args | grep -c -- "[f]req-embedding/scripts/train.py\|[t]rain_forecasting_head.py"' \
             2>/dev/null | tail -1)
    [ "${busy:-0}" -gt 0 ] || idle="$idle $label($id,idle)"
  done < <(awk '/cf373r2-/{print $1, $2, $5}' <<<"$st")

  printf '[%s] boxes=%s rate=$%s/h credit=%s done=%s missing=%s%s\n' \
    "$(date '+%m-%d %H:%M')" "${n_box:-0}" "${rate:-0.00}" "${cred:-?}" \
    "${done_n:-?}" "${miss_n:-?}" \
    "${idle:+  IDLE:$idle}"
  sleep "$INTERVAL"
done
