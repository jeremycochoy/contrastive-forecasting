#!/bin/bash
# #390 — one watchdog for the WHOLE pipeline, not one per wave.
#
# `monitor.sh` guards a single wave and exits with that wave's orchestrator.
# `pipeline.sh` runs three waves plus three eval stages back to back over
# several days, and CLAUDE.md § Remote Machine Monitoring wants a sync loop
# alive for the FULL duration of the run. This one watches `pipeline.pid` and
# exits only once the pipeline is gone. Every tick it
#
#   * copies <run>_losses.csv and <run>_attn_amplitude.csv from runs/ into
#     sync/<arm>/ (atomic: .tmp then mv), so a reader never catches a
#     half-written line,
#   * records each arm's newest checkpoint, last logged step and sps,
#   * reports a NaN with the STEP it happened at, so a wave-1 NaN is not
#     re-reported as news during waves 2 and 3,
#   * records the eval stage's progress: configs done, per cell.
#
# Usage:
#   WT=/home/jupyter/wt-cf-390-train \
#     nohup setsid bash watchdog.sh > /dev/null 2>&1 &
#
# Verify it by `ls`-ing sync/ after one tick, never by reading its log.
set -uo pipefail

INTERVAL="${1:-900}"                       # 15-min ticks (CLAUDE.md)
STARTUP_TICKS="${STARTUP_TICKS:-4}"        # ticks to wait for pipeline.pid
WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

OUT="$WT/experiments/2026-08-01_lalign_teacher"
RUNS="$OUT/runs"; SYNC="$OUT/sync"; RES="$OUT/results"
EVALS="$OUT/eval_gm_mase"
LOG="$RES/watchdog.log"
PIPE_PIDFILE="$RES/pipeline.pid"
mkdir -p "$SYNC" "$RES" "$RUNS"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >>"$LOG"; }

copy_atomic(){ # src dst min_bytes
  local src="$1" dst="$2" min="$3"
  [ -f "$src" ] || return 1
  [ "$(stat -c%s "$src")" -ge "$min" ] || return 1
  cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
}

pipeline_alive(){
  local pid; pid="$(cat "$PIPE_PIDFILE" 2>/dev/null)" || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

say "watchdog start — WT=$WT interval=${INTERVAL}s arms=${#CF390_ARMS[@]}"
idle_ticks=0
seen_pipeline=0
while true; do
  ts="$(date +%m-%d-%H:%M)"
  alive_pipe=no; pipeline_alive && { alive_pipe=yes; seen_pipeline=1; }
  training=0
  for arm in "${CF390_ARMS[@]}"; do
    name="$(bb_name "$arm")" || continue
    tlog="$RES/run_${name}.log"
    mkdir -p "$SYNC/$arm"
    # Per-class floors, never one blanket number: the losses CSV grows a few
    # hundred bytes per logged step, the attention CSV a little less.
    for f in "$RUNS/${name}"*_losses.csv; do
      [ -e "$f" ] || continue
      copy_atomic "$f" "$SYNC/$arm/$(basename "$f")" 1024
    done
    for f in "$RUNS/${name}"*_attn_amplitude.csv; do
      [ -e "$f" ] || continue
      copy_atomic "$f" "$SYNC/$arm/$(basename "$f")" 256
    done
    ck=$(ls -t "$RUNS/${name}"*_*k.pth 2>/dev/null | grep -v optimizer | head -1)
    last="$(grep -oE '^\[ *[0-9]+\]' "$tlog" 2>/dev/null | tail -1 | tr -dc '0-9')"
    sps="$(grep -oE '[0-9.]+ sps' "$tlog" 2>/dev/null | tail -1)"
    up=$(pgrep -f -- "--run-name $name" >/dev/null && echo yes || echo no)
    [ "$up" = yes ] && training=$((training + 1))
    # Report the STEP the NaN happened at, not a count over an append-only
    # log: a wave-1 NaN must not read as fresh news in waves 2 and 3.
    nan_step="$(grep -oE 'NaN/Inf DETECTED at step [0-9]+' "$tlog" 2>/dev/null \
                | tail -1 | grep -oE '[0-9]+$')"
    line="$ts $arm step=${last:-0} ${sps:-?} training=$up ck=$(basename "${ck:-<none>}")"
    [ -n "$nan_step" ] && line="$line NAN_AT_STEP=$nan_step"
    echo "$line" >>"$LOG"
  done
  # Eval-stage progress, straight off the CSVs the eval writes.
  for d in "$EVALS"/*/; do
    [ -d "$d" ] || continue
    csv="$d/gift/all_results.csv"
    [ -f "$csv" ] || continue
    echo "$ts eval $(basename "$d") configs=$(( $(wc -l < "$csv") - 1 ))/97" >>"$LOG"
  done
  echo "$ts pipeline=$alive_pipe trainers=$training" >>"$LOG"

  if [ "$alive_pipe" = no ]; then
    if [ "$seen_pipeline" -eq 1 ]; then
      say "pipeline process gone — watchdog exiting"
      exit 0
    fi
    idle_ticks=$((idle_ticks + 1))
    say "no pipeline PID yet ($idle_ticks/$STARTUP_TICKS)"
    if [ "$idle_ticks" -ge "$STARTUP_TICKS" ]; then
      say "*** no pipeline within $STARTUP_TICKS ticks — watchdog exiting, RUN UNGUARDED ***"
      exit 1
    fi
  else
    idle_ticks=0
  fi
  sleep "$INTERVAL"
done
