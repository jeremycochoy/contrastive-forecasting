#!/bin/bash
# #390 — 15-minute watchdog on the training host, plus a light-artefact copy.
#
# This runs ON the training host, so its runs/ → sync/ copy is same-disk and
# is NOT machine-death protection. Surviving the machine is sync_loop.sh's
# job: it runs on the box that owns the permanent checkout and pulls off
# host. What this adds is the watchdog, plus a stable point-in-time copy of
# the two CSVs — the trainer appends to them as it goes, so a reader can
# otherwise catch a half-written line. Every tick this
#
#   * copies <run>_losses.csv and <run>_attn_amplitude.csv from runs/ into
#     the experiment's sync/ (atomic: .tmp then mv),
#   * logs each arm's last step, sps and liveness to results/monitor.log,
#   * shouts on NaN, and on a trainer that died before the wave's budget.
#
# CLAUDE.md § Remote Machine Monitoring requires a sync loop for the FULL
# duration of every run, short or long. Start this alongside orchestrate.sh:
#
#   WT=$HOME/workspaces/contrastive-forecasting WAVE=1 \
#     nohup setsid bash monitor.sh > /dev/null 2>&1 &
#
# Verify it by `ls`-ing sync/ after one tick, never by reading monitor.log —
# a missing failure line can mean the pattern never matched.
set -uo pipefail

INTERVAL="${1:-900}"                       # 15-min ticks (CLAUDE.md)
# Ticks to wait for the FIRST trainer before giving up. The monitor is
# started BEFORE orchestrate.sh (so no part of the wave is unguarded), so
# on tick 1 no arm is up yet and "nothing alive" must not mean "done".
STARTUP_TICKS="${STARTUP_TICKS:-4}"
WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
case "$WT" in
  /tmp/*|/tmp)
    echo "ABORT: WT=$WT is under /tmp — refusing." >&2
    exit 2
    ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

WAVE="${WAVE:-1}"
case "$WAVE" in
  1|2|3) : ;;
  *) echo "ABORT: WAVE must be 1, 2 or 3; got '$WAVE'" >&2; exit 2 ;;
esac
BUDGET="${CF390_WAVE_TARGET_STEPS[$WAVE]}"   # this wave's step budget

OUT="$WT/experiments/2026-08-01_lalign_teacher"
RUNS="$OUT/runs"
SYNC="$OUT/sync"
LOG="$OUT/results/monitor.log"
ARMS_STR="${ARMS:-${CF390_ARMS[*]}}"
read -r -a ARMS <<< "$ARMS_STR"
mkdir -p "$SYNC" "$OUT/results"

copy_atomic() {   # src dst min_bytes
  local src="$1" dst="$2" min="$3"
  [ -f "$src" ] || return 1
  [ "$(stat -c%s "$src")" -ge "$min" ] || return 1
  cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
}

# orchestrate.sh runs the 10 cells as 5 sequential pairs, so between two
# phases nothing is training and the wave is nowhere near over. The
# orchestrator's own PID is the signal that says which of the two it is.
ORCH_PIDFILE="${ORCH_PIDFILE:-$OUT/results/orchestrate_wave${WAVE}.pid}"
orchestrator_alive() {
  local pid
  pid="$(cat "$ORCH_PIDFILE" 2>/dev/null)" || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null   # a stale PID file reads as gone
}

echo "[$(date '+%m-%d %H:%M:%S')] monitor start — wave=$WAVE budget=$BUDGET arms=${#ARMS[@]}" >>"$LOG"

declare -A was_alive=()   # per arm: has THIS monitor seen it training?
seen_alive=0     # has ANY arm ever been up? Guards the tick-1 exit below.
seen_orch=0      # has the orchestrator ever been up?
idle_ticks=0
while true; do
  ts="$(date +%m-%d-%H:%M)"
  running=0
  orch=no; orchestrator_alive && orch=yes
  [ "$orch" = yes ] && seen_orch=1
  for arm in "${ARMS[@]}"; do
    name="$(bb_name "$arm")" || continue
    tlog="$OUT/results/run_${name}.log"
    mkdir -p "$SYNC/$arm"
    # Losses CSV grows a few hundred bytes per logged step; the attention
    # amplitude CSV likewise. Per-class floors, never one blanket number.
    copy_atomic "$RUNS/${name}_losses.csv" \
                "$SYNC/$arm/${name}_losses.csv"         1024
    copy_atomic "$RUNS/${name}_attn_amplitude.csv" \
                "$SYNC/$arm/${name}_attn_amplitude.csv"  256
    last="$(grep -oE '^\[ *[0-9]+\]' "$tlog" 2>/dev/null | tail -1 | tr -dc '0-9')"
    sps="$(grep -oE '[0-9.]+ sps' "$tlog" 2>/dev/null | tail -1)"
    alive=$(pgrep -f "run-name $name" >/dev/null && echo yes || echo no)
    [ "$alive" = yes ] && { running=$((running + 1)); was_alive[$arm]=1; }
    # `grep -c` exits 1 on zero matches but still prints "0", so swallow the
    # status rather than appending an `echo 0` (which produced "0\n0" and a
    # false NaN alert on #388's first tick). On a log that does not exist it
    # prints nothing at all, and an empty count is not "0" — that fired the
    # alert for every arm still waiting for its phase.
    nan=$(grep -c 'NaN/Inf DETECTED' "$tlog" 2>/dev/null || true)
    nan="${nan:-0}"
    echo "$ts $arm step=${last:-0} ${sps:-?} alive=$alive nan=$nan" >>"$LOG"
    [ "$nan" != 0 ] && echo "$ts *** NaN in $arm — cell dropped per the issue's stop rule ***" >>"$LOG"
    # run_arm.sh APPENDS to this log across waves, so before an arm starts
    # its wave-2 run the last step in it is wave 1's endpoint. Only an arm
    # this monitor has actually seen training can have died.
    if [ "$alive" = no ] && [ "${was_alive[$arm]:-0}" = 1 ] \
       && [ -n "$last" ] && [ "$last" -lt "$BUDGET" ]; then
      echo "$ts *** $arm died at step $last (wave-$WAVE budget $BUDGET) ***" >>"$LOG"
    fi
  done
  # "Nothing training" is true at every phase boundary — 4 times per wave,
  # and for the whole of a re-run whose arms all skip in seconds. The wave
  # is over only once the ORCHESTRATOR is gone too. Exiting early leaves the
  # rest of the wave unguarded, which is the silent no-monitor failure.
  if [ "$running" -eq 0 ] && [ "$orch" = no ]; then
    if [ "$seen_alive" -eq 1 ] || [ "$seen_orch" -eq 1 ]; then
      echo "$ts all arms stopped, orchestrator gone — monitor exiting" >>"$LOG"
      exit 0
    fi
    # Nothing has ever run and no orchestrator ever showed up. The launch
    # order is monitor first, then orchestrate, so allow a few ticks.
    idle_ticks=$((idle_ticks + 1))
    echo "$ts no arm up yet ($idle_ticks/$STARTUP_TICKS)" >>"$LOG"
    if [ "$idle_ticks" -ge "$STARTUP_TICKS" ]; then
      echo "$ts *** no arm started within $STARTUP_TICKS ticks — monitor" \
           "exiting, THE WAVE IS UNGUARDED ***" >>"$LOG"
      exit 1
    fi
  else
    [ "$running" -gt 0 ] && seen_alive=1
    [ "$running" -eq 0 ] && \
      echo "$ts no arm training, orchestrator up — between phases" >>"$LOG"
    idle_ticks=0
  fi
  sleep "$INTERVAL"
done
