#!/bin/bash
# #390 — 15-minute watchdog on the training host, plus a light-artefact copy.
#
# The waves run on elisa, where the runs are local and there is no scp proxy
# to sync through. What still has to be guarded is what sync_loop guards on
# a remote: the machine can die, and the small CSVs are the irreplaceable
# part (a checkpoint can be retrained; a loss curve cannot be recovered).
# Every tick this
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

WAVE="${WAVE:-1}"
case "$WAVE" in
  1) BUDGET=40000 ;;
  2) BUDGET=100000 ;;
  3) BUDGET=200000 ;;
  *) echo "ABORT: WAVE must be 1, 2 or 3; got '$WAVE'" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

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

echo "[$(date '+%m-%d %H:%M:%S')] monitor start — wave=$WAVE budget=$BUDGET arms=${#ARMS[@]}" >>"$LOG"

seen_alive=0     # has ANY arm ever been up? Guards the tick-1 exit below.
idle_ticks=0
while true; do
  ts="$(date +%m-%d-%H:%M)"
  running=0
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
    [ "$alive" = yes ] && running=$((running + 1))
    # `grep -c` exits 1 on zero matches but still prints "0", so swallow the
    # status rather than appending an `echo 0` (which produced "0\n0" and a
    # false NaN alert on #388's first tick).
    nan=$(grep -c 'NaN/Inf DETECTED' "$tlog" 2>/dev/null || true)
    echo "$ts $arm step=${last:-0} ${sps:-?} alive=$alive nan=$nan" >>"$LOG"
    [ "$nan" != 0 ] && echo "$ts *** NaN in $arm — cell dropped per the issue's stop rule ***" >>"$LOG"
    if [ "$alive" = no ] && [ -n "$last" ] && [ "$last" -lt "$BUDGET" ]; then
      echo "$ts *** $arm died at step $last (wave-$WAVE budget $BUDGET) ***" >>"$LOG"
    fi
  done
  # "Nothing alive" means "the wave finished" only AFTER something has run.
  # On tick 1 the orchestrator has not spawned python yet, and exiting there
  # would leave the whole wave unguarded — the silent no-monitor failure.
  if [ "$running" -gt 0 ]; then
    seen_alive=1
  elif [ "$seen_alive" -eq 1 ]; then
    echo "$ts all arms stopped — monitor exiting" >>"$LOG"
    exit 0
  else
    idle_ticks=$((idle_ticks + 1))
    echo "$ts no arm up yet ($idle_ticks/$STARTUP_TICKS)" >>"$LOG"
    if [ "$idle_ticks" -ge "$STARTUP_TICKS" ]; then
      echo "$ts *** no arm started within $STARTUP_TICKS ticks — monitor" \
           "exiting, THE WAVE IS UNGUARDED ***" >>"$LOG"
      exit 1
    fi
  fi
  sleep "$INTERVAL"
done
