#!/bin/bash
# #328 — drive both q-heads. Serial on one GPU, or parallel if two are given.
#   run_downstream.sh <gpu2L> [gpu6L] [evalsets]
# One GPU:   run_downstream.sh 0
# Two GPUs:  run_downstream.sh 0 1
set -uo pipefail
G2="${1:?gpu for 2L}"; G6="${2:-$G2}"; SETS="${3:-triage full}"
HERE="$(cd "$(dirname "$0")" && pwd)"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [run-dl] $*"; }
if [ "$G2" = "$G6" ]; then
  log "serial on GPU $G2: 2L then 6L"
  bash "$HERE/downstream_triplet.sh" 2 "$G2" "$SETS"
  bash "$HERE/downstream_triplet.sh" 6 "$G6" "$SETS"
else
  log "parallel: 2L->GPU $G2, 6L->GPU $G6"
  bash "$HERE/downstream_triplet.sh" 2 "$G2" "$SETS" &
  p2=$!
  bash "$HERE/downstream_triplet.sh" 6 "$G6" "$SETS" &
  p6=$!
  wait $p2; r2=$?; wait $p6; r6=$?
  log "2L rc=$r2  6L rc=$r6"
  [ $r2 -eq 0 ] && [ $r6 -eq 0 ] || exit 1
fi
log "both heads done"
