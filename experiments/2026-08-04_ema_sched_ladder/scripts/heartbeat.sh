#!/bin/bash
# #393 — one event per line, plus an hourly heartbeat whether or not
# anything happened.
#
# Silence from an event-only watcher is ambiguous: a fleet climbing quietly
# and a watcher that died look identical. The heartbeat resolves that, and
# CLAUDE.md requires one for the whole life of a remote run.
#
# Events, in the order they matter:
#   SCORE   a GM-Relative MASE landed — the study's only output
#   DEC     an extend-rule branch fired
#   BROKER  an eval finished on elisa, or failed
#   DEAD    a ladder driver that was alive is not any more
#   ERR     a crash signature in a run log
#
# Reads the local sync roots rather than sshing the boxes: the six sync
# loops already pull ladder.csv, decisions.csv and the run logs every 15
# minutes, and a watcher that opens 6 ssh connections a minute competes
# with them through the same proxy.
set -uo pipefail

EXP="${EXP:-/tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder}"
STATE="${HEARTBEAT_STATE:-/tmp/cf393_heartbeat.state}"
POLL="${HEARTBEAT_POLL:-120}"
BEAT="${HEARTBEAT_EVERY:-3600}"
ROOTS=("$EXP" "$HOME"/cf393_sync{,_b,_c,_d,_e,_f}/2026-08-04_ema_sched_ladder)

: >>"$STATE"
# Only lines not seen before reach stdout. Sorting keeps the file usable as
# a `grep -f` pattern set as it grows.
emit() {
  local line
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    grep -qxF -- "$line" "$STATE" 2>/dev/null && continue
    printf '%s\n' "$line"
    printf '%s\n' "$line" >>"$STATE"
  done
}

# `grep -c` and `pgrep -c` both PRINT 0 and EXIT 1 when they match nothing,
# so the usual `|| echo 0` appends a second zero and the heartbeat line
# breaks in two. Take the first line and ignore the status.
count() { local n; n="$("$@" 2>/dev/null | head -1)"; printf '%s' "${n:-0}"; }
drivers_alive() { count pgrep -fc "[l]adder\.py --cells"; }

last_beat=0
while :; do
  {
    for r in "${ROOTS[@]}"; do
      tag="$(basename "$(dirname "$r")")"; [ "$tag" = "experiments" ] && tag=elisa
      [ -f "$r/results/ladder.csv" ] && \
        awk -F, -v t="$tag" 'NR>1 && $8!="" {printf "SCORE %s %s bb%dk %s GM-Rel-MASE=%s\n", t, $1, $4/1000, $5, $8}' \
          "$r/results/ladder.csv"
      [ -f "$r/results/decisions.csv" ] && \
        awk -F, -v t="$tag" 'NR>1 {printf "DEC %s %s @%s %s extend=%s heads=%s\n", t, $1, $2, $3, $4, $5}' \
          "$r/results/decisions.csv"
      grep -hoE "Traceback|CUDA error[^\"]{0,40}|out of memory|Killed|AcceleratorError" \
        "$r"/results/run_cf393_*.log 2>/dev/null | sed "s|^|ERR $tag |" | sort -u
    done
    grep -hoE "\[.*\] \[broker\] .*(DONE score=|rc=[0-9]+ —|pull failed|upload failed|swap failed).*" \
      "$EXP/results/eval_broker.log" 2>/dev/null | sed 's|^|BROKER |'
  } 2>/dev/null | emit

  now=$(date +%s)
  if [ $(( now - last_beat )) -ge "$BEAT" ]; then
    last_beat=$now
    n_scores=$(count grep -c '^SCORE ' "$STATE")
    n_eval=$(count pgrep -fc "[e]val_gift_eval_official")
    printf 'HEARTBEAT %s drivers=%s scores=%s eval_procs=%s slots=%s load=%s credit=%s\n' \
      "$(date '+%m-%d %H:%M')" "$(drivers_alive)" "$n_scores" "$n_eval" \
      "$(ls /tmp/cf393_evalslots 2>/dev/null | wc -l)" \
      "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')"
  fi
  sleep "$POLL"
done
