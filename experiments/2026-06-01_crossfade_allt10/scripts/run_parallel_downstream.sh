#!/bin/bash
# #325 — parallel downstream: 2L q-head on GPU0, 6L q-head on GPU1 (both freed after the
# backbone finished). Each: train 30k q-head + GIFT-Eval triage+full. Idempotent (per-cell
# skips). Halves downstream wall-clock vs the serial run.sh path. Touches .all_done when both
# cells finish, so the monitor detects completion.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-01_crossfade_allt10}"
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$RES/parallel_downstream.log"; }
log "PARALLEL DOWNSTREAM start: 2L->GPU0, 6L->GPU1"
bash "$HERE/downstream_crossfade.sh" 2 0 "triage full" >>"$RES/cell_2L.log" 2>&1 &
P2=$!
bash "$HERE/downstream_crossfade.sh" 6 1 "triage full" >>"$RES/cell_6L.log" 2>&1 &
P6=$!
wait $P2; r2=$?
wait $P6; r6=$?
log "PARALLEL DOWNSTREAM done: 2L rc=$r2, 6L rc=$r6"
[ $r2 -eq 0 ] && [ $r6 -eq 0 ] && touch "$RES/.all_done" && log "touched .all_done"
