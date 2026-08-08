#!/bin/bash
# #373 — run one box's backbone jobs, one after another.
#
# Runs ON the vast.ai box, started by launch_box.sh. Each job is one
# (cell, k) pair trained to one stop. A vast.ai box comes up in
# `Exclusive_Process` compute mode and the container cannot change it, so
# only one CUDA context exists at a time: jobs are serial here by the
# hardware, not by choice.
#
# Usage (on the box):  bash queue_remote.sh <cell>:<k>:<steps> [...]
#   e.g.               bash queue_remote.sh B5:0:40000 B5:3:40000
set -uo pipefail

WT=/root/cf
HERE="$WT/reports/2026-08-08_rollout_depth/scripts"
RES="$WT/reports/2026-08-08_rollout_depth/results"
mkdir -p "$RES"
export CF373_ROOT=/root/cf373_runs

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [queue] $*" | tee -a "$RES/queue.log"; }

[ $# -gt 0 ] || { log "ABORT: no jobs"; exit 2; }
log "box starts with $# job(s): $*"

for job in "$@"; do
  IFS=: read -r cell k steps <<<"$job"
  [ -n "${steps:-}" ] || { log "ABORT: bad job '$job', want <cell>:<k>:<steps>"; exit 2; }

  row="$(awk -F'\t' -v c="$cell" '$1==c {print; exit}' "$HERE/cells.tsv")"
  [ -n "$row" ] || { log "ABORT: no cell '$cell' in cells.tsv"; exit 2; }
  launcher=$(cut -f3 <<<"$row"); arg=$(cut -f4 <<<"$row")

  args=("$arg")
  case "$launcher" in run_leg_k.sh) args+=("$steps") ;; esac

  log "START $cell k=$k -> ${steps} steps via $launcher $arg"
  K="$k" TARGET_STEPS="$steps" FINAL_STEPS=200000 \
  SAVE_EVERY=20000 EXTRA_SAVES="$steps" \
  WT="$WT" BB_GPU=0 RUNS="$CF373_ROOT" CF373_RUNS="$CF373_ROOT" \
    bash "$HERE/$launcher" "${args[@]}"
  rc=$?
  log "END   $cell k=$k rc=$rc"
  # A failed job does not take the box down: the next job is a different
  # cell and its artefacts are worth the remaining GPU time. The sync loop
  # brings the log back either way.
done

log "box queue drained"
touch "$RES/QUEUE_DONE"
