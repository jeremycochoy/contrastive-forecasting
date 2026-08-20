#!/bin/bash
# #404 — one liveness line for the round that runs now.
#
# A notification that never fires and a run that never ends look the same from
# here. On 2026-06-11 a box spent a night degraded, technically "running", and
# the money drained. So this probe reads the things that MOVE, not the things
# that merely exist:
#
#   driver     the round 3c pid, from `results/round3c.pid`.
#   card       GPU utilization, memory in use, compute apps, off the box.
#   progress   the last step the live job wrote, and the step before it, so a
#              reader sees a counter that advances and not a file that is there.
#   scores     which score files exist.
#   spend      what vast.ai has billed this instance.
#
# Usage:  bash scripts/heartbeat.sh          # one line, then exit
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
R="$STUDY/results"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

INSTANCE=""; HOST=""; PORT=""
# shellcheck disable=SC1090
[ -s "$R/round3.env" ] && . "$R/round3.env"

drv="down"
if [ -s "$R/round3c.pid" ] && ps -p "$(cat "$R/round3c.pid")" >/dev/null 2>&1; then
  drv="up($(cat "$R/round3c.pid"))"
fi

card="?"; prog="?"
if [ -n "$HOST" ] && [ -n "$PORT" ]; then
  card="$(timeout 90 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ' | tr '\n' ' '; \
     echo -n \"apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)\"" 2>/dev/null)"
  # The step counter of whatever runs now: the newest losses CSV under the run
  # root. Two rows, so a reader sees the counter advance between two probes.
  prog="$(timeout 90 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "csv=\$(ls -t /root/cf404_runs/*/*/*/*_losses.csv /root/cf404_runs/*/eval/*/*_losses.csv 2>/dev/null | head -1); \
     [ -n \"\$csv\" ] && echo \"\$(basename \$csv | cut -c1-46) step=\$(tail -1 \$csv | cut -d, -f1)\"" 2>/dev/null)"
fi

scores=""
for f in "$R"/score_*_bb40k_h30k_student.txt; do
  [ -s "$f" ] || continue
  a="$(basename "$f" .txt)"; a="${a#score_}"; a="${a%%_bb*}"
  scores="$scores $a=$(tr -d ' \t\r\n' <"$f")"
done

spend="$(timeout 120 vastrun-status 2>/dev/null \
  | awk -v id="${INSTANCE:-none}" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print v }')"

echo "[$(date '+%m-%d %H:%M')] #404 driver=$drv card=${card:-unreachable}" \
     "| ${prog:-no csv} | scores:${scores:- none} | spend=${spend:-gone}"
