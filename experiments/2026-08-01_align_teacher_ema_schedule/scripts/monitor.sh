#!/bin/bash
# #388 — 15-minute watchdog + light-artefact copy.
#
# The runs are local to elisa, so there is no scp proxy to sync through.
# What still has to be guarded is the same thing sync_loop guards on a
# remote: the machine can die, and the small CSVs are the irreplaceable
# part. Every tick this
#
#   * copies <run>_losses.csv and <run>_latent_drift.csv from the durable
#     run dir into the experiment's sync/ (atomic: .tmp then mv),
#   * logs each arm's last step, sps and liveness to results/monitor.log,
#   * shouts on NaN or on a dead trainer that never reached the budget.
#
# Usage: WT=/tmp/contrastive-forecasting-388 monitor.sh [interval_s]
set -uo pipefail

INTERVAL="${1:-900}"
WT="${WT:?WT (worktree root) must be set}"
OUT="$WT/experiments/2026-08-01_align_teacher_ema_schedule"
RUNS_ROOT="${RUNS_ROOT:-/home/jupyter/checkpoints_backup/cf-388}"
SYNC="$OUT/sync"
LOG="$OUT/results/monitor.log"
ARMS=(align_teacher_a09 align_teacher_sched pred_moco_sched rep_moco_sched)
mkdir -p "$SYNC" "$OUT/results"

copy_atomic() {   # src dst min_bytes
  local src="$1" dst="$2" min="$3"
  [ -f "$src" ] || return 1
  [ "$(stat -c%s "$src")" -ge "$min" ] || return 1
  cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
}

while true; do
  ts="$(date +%m-%d-%H:%M)"
  running=0
  for arm in "${ARMS[@]}"; do
    name="ats_${arm}"
    dir="$RUNS_ROOT/$arm"
    tlog="$OUT/results/run_${name}.log"
    mkdir -p "$SYNC/$arm"
    # losses CSV grows ~280 B/step; the drift CSV a few hundred bytes per probe.
    copy_atomic "$dir/${name}_losses.csv"       "$SYNC/$arm/${name}_losses.csv"       1024
    copy_atomic "$dir/${name}_latent_drift.csv" "$SYNC/$arm/${name}_latent_drift.csv"  64
    last="$(grep -oE '^\[ *[0-9]+\]' "$tlog" 2>/dev/null | tail -1 | tr -dc '0-9')"
    sps="$(grep -oE '[0-9.]+ sps' "$tlog" 2>/dev/null | tail -1)"
    alive=$(pgrep -f "run-name $name" >/dev/null && echo yes || echo no)
    [ "$alive" = yes ] && running=$((running + 1))
    nan=$(grep -c 'NaN/Inf DETECTED' "$tlog" 2>/dev/null || echo 0)
    echo "$ts $arm step=${last:-0} ${sps:-?} alive=$alive nan=$nan" >>"$LOG"
    [ "$nan" != 0 ] && echo "$ts *** NaN in $arm ***" >>"$LOG"
    if [ "$alive" = no ] && [ -n "$last" ] && [ "$last" -lt 100000 ]; then
      echo "$ts *** $arm died at step $last (budget 100000) ***" >>"$LOG"
    fi
  done
  if [ "$running" -eq 0 ]; then
    echo "$ts all arms stopped — monitor exiting" >>"$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
