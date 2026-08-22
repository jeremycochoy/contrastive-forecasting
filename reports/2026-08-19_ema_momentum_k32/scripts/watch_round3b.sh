#!/bin/bash
# #404 — the watcher of round 3b. One line per event, on stdout.
#
# TWO STREAMS, ONE OUTPUT.
#
#   the milestones  every new line of `results/round3b.out` that a reader must
#                   act on: a score, a verdict, a failure, the watchdog, the
#                   teardown.
#   the heartbeat   every HEARTBEAT_SECS, an ACTIVE probe of the box. A driver
#                   that hangs prints nothing, and silence reads the same as
#                   progress. On 2026-06-11 a vast.ai box spent a night
#                   degraded while a session waited on events alone.
#
# The probe reads four things that a stalled run cannot fake: the driver
# process, the step counter, the GPU, and the spend. It says STALLED when the
# step counter has not moved since the probe before it.
#
# Usage:  bash scripts/watch_round3b.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="${OUT:-$CF404_RESULTS/round3b.out}"
PIDF="${PIDF:-$CF404_RESULTS/round3b.pid}"
ENVF="${ENVF:-$CF404_RESULTS/round3.env}"
HEARTBEAT_SECS="${HEARTBEAT_SECS:-3600}"
POLL="${POLL:-60}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

# shellcheck disable=SC1090
[ -s "$ENVF" ] && . "$ENVF"
rsh(){ timeout 60 ssh "${SSH_OPTS[@]}" -p "${PORT:-22}" "root@${HOST:-nowhere}" "$@" 2>/dev/null; }

# The milestones a reader acts on. Everything else stays in the log.
KEEP='SCORE |: DONE|VERIFIED|FAILED|ABORT|WATCHDOG|teardown:|STEP RATE|^\[.*====|ROUND 3B DONE'

seen=0
[ -f "$OUT" ] && seen="$(grep -c '^' "$OUT")"
last_step=""; waited=0

probe(){
  local pid alive step sps gpu spent arm run
  pid="$(cat "$PIDF" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then alive=yes; else alive=no; fi
  arm="$(grep -oE '==== [a-z0-9]+ :' "$OUT" 2>/dev/null | tail -1 | awk '{print $2}')"
  if [ -n "$arm" ]; then
    run="/root/cf/reports/$(basename "$CF404_STUDY")/results/run_$(cf404_run_name "$arm").log"
    step="$(rsh "grep -oE '^\[ *[0-9]+\]' $run 2>/dev/null | tail -1 | tr -d '[] '")"
    sps="$(rsh "grep -hoE '[0-9.]+ sps' $run 2>/dev/null | tail -1")"
  fi
  gpu="$(rsh "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" | tr -d '\n')"
  spent="$(timeout 120 vastrun-status 2>/dev/null \
    | awk -v id="${INSTANCE:-none}" '$1 == id { for (i = 1; i <= NF; i++) if ($i ~ /^\$/) v = $i; print v }')"
  if [ -z "$gpu" ] && [ -z "$spent" ]; then
    printf 'HEARTBEAT driver=%s arm=%s box=GONE\n' "$alive" "${arm:-none}"
  elif [ -n "$step" ] && [ "$step" = "$last_step" ]; then
    printf 'HEARTBEAT STALLED driver=%s arm=%s step=%s (no move) gpu=%s spent=%s\n' \
      "$alive" "${arm:-none}" "$step" "${gpu:-?}" "${spent:-?}"
  else
    printf 'HEARTBEAT driver=%s arm=%s step=%s %s gpu=%s spent=%s\n' \
      "$alive" "${arm:-none}" "${step:-?}" "${sps:-?}" "${gpu:-?}" "${spent:-?}"
  fi
  last_step="$step"
}

probe
while true; do
  sleep "$POLL"; waited=$(( waited + POLL ))
  if [ -f "$OUT" ]; then
    now="$(grep -c '^' "$OUT")"
    if [ "$now" -gt "$seen" ]; then
      tail -n +$(( seen + 1 )) "$OUT" | grep -E "$KEEP"
      seen="$now"
    fi
  fi
  if [ "$waited" -ge "$HEARTBEAT_SECS" ]; then probe; waited=0; fi
  # The driver is gone and the log is drained, so there is nothing left to say.
  pid="$(cat "$PIDF" 2>/dev/null)"
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    grep -qE 'ROUND 3B DONE|ABORT' "$OUT" 2>/dev/null && { echo "WATCHER: the driver has ended"; exit 0; }
  fi
done
