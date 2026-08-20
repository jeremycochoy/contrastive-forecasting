#!/bin/bash
# #404 — one liveness line for the round that runs now.
#
# A notification that never fires and a run that never ends look the same from
# here. On 2026-06-11 a box spent a night degraded, technically "running", and
# the money drained. So this probe reads the things that MOVE, not the things
# that merely exist:
#
#   driver     the round's pid, from `results/<round>.pid`.
#   card       GPU utilization, memory in use, compute apps, off the box.
#   progress   the last step EVERY live job wrote, so a reader sees a counter
#              that advances and not a file that is there. Round 4 runs two
#              lanes at a time, so one counter is not enough.
#   scores     which score files exist.
#   spend      what vast.ai has billed this instance.
#
# The round is a knob, so this probe follows the study instead of naming one
# driver. `ROUND=round3c bash scripts/heartbeat.sh` reads the round before it.
#
# Usage:  bash scripts/heartbeat.sh          # one line, then exit
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
R="$STUDY/results"
ROUND="${ROUND:-round4}"
ENVF="${ROUND_ENV:-$R/$ROUND.env}"
PIDF="${ROUND_PID:-$R/$ROUND.pid}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

INSTANCE=""; HOST=""; PORT=""
# shellcheck disable=SC1090
[ -s "$ENVF" ] && . "$ENVF"

drv="down"
if [ -s "$PIDF" ] && ps -p "$(cat "$PIDF")" >/dev/null 2>&1; then
  drv="up($(cat "$PIDF"))"
fi

card="?"; prog="?"
if [ -n "$HOST" ] && [ -n "$PORT" ]; then
  card="$(timeout 90 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ' | tr '\n' ' '; \
     echo -n \"apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)\"" 2>/dev/null)"
  # EVERY losses CSV under the run root, newest four, with its last step. Two
  # lanes run at a time, so a probe that reads one file calls a dead lane live.
  #
  # The arm name may hold an UNDERSCORE (`r100_09`), so neither pattern may
  # stop at one. The backbone pattern anchors on `_cf373k<N>_` to its left and
  # the head pattern anchors on `_bb<N>k_` to its right. A class of
  # `[a-z0-9]+` printed the whole file name for the backbone and `head_r100`
  # for the head.
  prog="$(timeout 90 ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" \
    "for csv in \$(ls -t /root/cf404_runs/*/*/*/*_losses.csv /root/cf404_runs/*/eval/*/*_losses.csv 2>/dev/null | head -4); do \
       printf '%s=%s ' \"\$(basename \$csv | sed -E 's/^.*_cf373k[0-9]+_(mean_[a-z0-9_]+)_losses.csv$/\1/; s/^qhead_(.*)_bb[0-9]+k_.*$/head_\1/')\" \"\$(tail -1 \$csv | cut -d, -f1)\"; \
     done" 2>/dev/null)"
fi

scores=""
for f in "$R"/score_*_bb40k_h30k_student.txt; do
  [ -s "$f" ] || continue
  a="$(basename "$f" .txt)"; a="${a#score_}"; a="${a%%_bb*}"
  scores="$scores $a=$(tr -d ' \t\r\n' <"$f")"
done

spend="$(timeout 120 vastrun-status 2>/dev/null \
  | awk -v id="${INSTANCE:-none}" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print v }')"

echo "[$(date '+%m-%d %H:%M')] #404 $ROUND driver=$drv card=${card:-unreachable}" \
     "| ${prog:-no csv} | scores:${scores:- none} | spend=${spend:-gone}"
