#!/bin/bash
# #373 round 3 — one line per hour on where the money and the queue stand.
#
# Usage: BOX_ID=<id> bash q_heartbeat.sh [interval seconds]
#
# CLAUDE.md § Remote Machine Monitoring: assume the machine can crash at any
# time. This writes what a crash would otherwise hide — spend, credit, the
# four cards' utilisation, and how many queue jobs are running against how
# many are left — so a gap in the file dates the failure.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
OUT="$RES/q_heartbeat.log"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
BOX_ID="${BOX_ID:?BOX_ID}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"
INTERVAL="${1:-3600}"
export PATH="$HOME/.local/bin:$PATH"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"

count(){ local s="$1" n=0 id
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
    [ "$(cat "$STATE/$id.state" 2>/dev/null || echo queued)" = "$s" ] && n=$(( n + 1 ))
  done; echo "$n"; }

while :; do
  ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  credit="$(cd "$VDIR" && timeout 90 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')"
  # The GPU column holds a space ("RTX 4090"), so Spent is field 8, not 7.
  # Read it by name off the header rather than by a counted offset.
  spent="$(cd "$VDIR" && timeout 90 vastrun-status 2>/dev/null \
           | awk -v b="$BOX_ID" '$1==b{for(i=NF;i>0;i--) if($i ~ /^\$/){print $i; exit}}')"
  rgpu="$(timeout 90 ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout=20 -p "$BOX_PORT" "root@$BOX_HOST" \
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null \
          | tr -d ' ' | paste -sd/ -)"
  lgpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
          | tr -d ' ' | paste -sd/ -)"
  lfree="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
          | tr -d ' ' | paste -sd/ -)"
  scores="$(ls "$RES"/score_*.txt 2>/dev/null | wc -l)"
  printf '%s credit=%s box_spent=%s box_gpu=%s elisa_gpu=%s elisa_free=%s run=%s done=%s fail=%s left=%s scores=%s\n' \
    "$ts" "${credit:-?}" "${spent:-?}" "${rgpu:-down}" "${lgpu:-?}" "${lfree:-?}" \
    "$(count running)" "$(count done)" "$(count failed)" "$(count queued)" "$scores" \
    | tee -a "$OUT"
  sleep "$INTERVAL"
done
