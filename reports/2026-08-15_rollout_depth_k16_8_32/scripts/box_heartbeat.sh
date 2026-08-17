#!/bin/bash
# #401 — one status line every half hour, and one immediately when something
# breaks. The mean arm runs on two machines for about a day.
#
# A watcher that fires only on completion misses a stall: a trainer that hangs
# without exiting sends no event, and a rented box that dies sends none either.
# CLAUDE.md § Remote Machine Monitoring assumes the machine can crash at any
# time, so this probes instead of waiting.
#
# It probes, it does not repair. Every line goes to stdout, one line per
# event, so a Monitor turns each into a notification.
#
# What it watches:
#   box       the instance is running, and what it has spent
#   legs      the step counter of each arm on the box moves
#   sync      a sync loop for this study's local root is alive
#   scores    a new GIFT-Eval score landed on elisa
#   elisa     the transitional elisa arms, while they still run
#
# Usage:
#   HOST=ssh2.vast.ai PORT=16048 CONTRACT=47976049 \
#     bash scripts/box_heartbeat.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

HOST="${HOST:?HOST must be set}"
PORT="${PORT:?PORT must be set}"
CONTRACT="${CONTRACT:?CONTRACT must be set}"
EVERY="${EVERY:-1800}"
TICK="${TICK:-300}"
BOX_RUNS="${CF401_BOX_RUNS}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)
rsh(){ ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "$@" </dev/null 2>/dev/null; }

say(){ echo "[$(date '+%m-%d %H:%M')] $*"
       echo "[$(date '+%m-%d %H:%M')] $*" >>"$CF401_RESULTS/heartbeat_box.log"; }

# The last step of one arm's newest losses CSV on the box. Empty when the box
# is unreachable, which is not the same as zero and must not read as a stall.
box_step(){  # <k>
  rsh "ls -1t $BOX_RUNS/k$1/$CF401_CELL/leg_*k/*_losses.csv 2>/dev/null \
       | head -1 | xargs -r tail -1 | cut -d, -f1" | tr -dc '0-9'
}

elisa_step(){  # <k>
  local f
  f="$(ls -1t "$(cf401_arm_root "$1")/$CF401_CELL"/leg_*k/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$f" ] && tail -1 "$f" | cut -d, -f1 | tr -dc '0-9'
}

n_scores(){ ls "$CF401_RESULTS"/score_*.txt 2>/dev/null | wc -l; }

declare -A last_step=()
prev_scores="$(n_scores)"
elapsed="$EVERY"   # report once at the start

while true; do
  # --- the box exists and bills -----------------------------------------------
  status="$(vastrun-status 2>/dev/null | awk -v id="$CONTRACT" '$1 == id {print}')"
  if [ -z "$status" ]; then
    say "ALERT box $CONTRACT is not in vastrun-status — the instance is gone"
    sleep "$TICK"; elapsed=$(( elapsed + TICK )); continue
  fi
  spent="$(printf '%s' "$status" | awk '{print $(NF-1)}')"
  rate="$(printf '%s' "$status" | awk '{print $(NF-2)}')"

  # --- each arm's step counter -------------------------------------------------
  line=""; stalled=""
  for k in $CF401_DEPTHS; do
    s="$(box_step "$k")"
    e="$(elisa_step "$k")"
    if [ -z "$s" ]; then
      line="$line k$k=box?"
    else
      if [ -n "${last_step[$k]:-}" ] && [ "$s" -le "${last_step[$k]}" ]; then
        stalled="$stalled k$k"
      fi
      last_step[$k]="$s"
      line="$line k$k=$s"
    fi
    [ -n "$e" ] && line="$line(elisa $e)"
  done

  # --- the sync loop, by working directory, not by its own log ----------------
  loops="$(cf401_sync_loops "$CF401_SYNC_DIR")"
  [ "$loops" -ge 1 ] || say "ALERT no sync_loop.sh runs for $CF401_SYNC_DIR — the box's checkpoints do not reach elisa"

  # --- a new score ------------------------------------------------------------
  now_scores="$(n_scores)"
  if [ "$now_scores" -gt "$prev_scores" ]; then
    say "SCORE $prev_scores -> $now_scores — $(ls -1t "$CF401_RESULTS"/score_*.txt | head -1 | xargs -r basename)"
    prev_scores="$now_scores"
  fi

  # A leg that neither moved nor finished in a whole tick is a stall. A leg
  # that ENDED is not: its arm's next leg starts under a new leg dir and the
  # step counter restarts from the stop below it, so the report names it and
  # the reader decides.
  [ -z "$stalled" ] || say "ALERT step counter did not move in ${TICK}s:$stalled"

  if [ "$elapsed" -ge "$EVERY" ]; then
    say "box $CONTRACT $rate spent $spent | steps$line | sync loops $loops | scores $now_scores"
    elapsed=0
  fi
  sleep "$TICK"; elapsed=$(( elapsed + TICK ))
done
