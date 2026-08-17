#!/bin/bash
# #401 — add the k = 32 200k leg to the collapse table as it lands.
#
# The k = 32 200k leg writes a backbone every 20,000 steps. Each new file
# is a subject `diag_collapse.py --all` picks up on its own, because the
# subject list comes from a directory walk. This script waits for a new
# file, re-runs the three probes on CPU, and rebuilds the table and plot.
#
# It touches no GPU and no score file. Phase 1 keeps its card.
#
# Usage:  bash scripts/watch_k32_200k.sh            # until 200k lands
#         WATCH_ONCE=1 bash scripts/watch_k32_200k.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
PY="${PY:-/home/jupyter/rnd/.venv/bin/python3}"
LEG="/home/jupyter/checkpoints_backup/cf-401/k32/arm6_v2_combab_alignS/leg_200k"
STEM="cf393_arm6_v2_combab_alignS_cf373k32"
LOG="$STUDY/results/diag/watch_k32_200k.log"

cd "$STUDY" || exit 2
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 watch] $*" | tee -a "$LOG"; }

rebuild(){
  log "probing, $(ls "$LEG"/${STEM}_*k.pth 2>/dev/null | wc -l) files in leg_200k"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
    $PY scripts/diag_collapse.py --all --out results/diag/collapse_all.csv \
    >results/diag/collapse_all.out 2>&1 || log "WARN collapse rc=$?"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
    $PY scripts/diag_time_rank.py --out results/diag/time_rank.csv \
    >results/diag/time_rank.out 2>&1 || log "WARN time_rank rc=$?"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
    $PY scripts/diag_scalar_readout.py --out results/diag/scalar_readout.csv \
    >results/diag/scalar_readout.out 2>&1 || log "WARN readout rc=$?"
  $PY scripts/diag_curve_state.py --out results/diag/curve_state.csv \
    >results/diag/curve_state.out 2>&1 || log "WARN curve rc=$?"
  $PY scripts/make_collapse_table.py \
    >results/diag/collapse_vs_score.out 2>&1 || log "WARN table rc=$?"
  $PY scripts/plot_collapse_vs_score.py >>"$LOG" 2>&1 || log "WARN plot rc=$?"
  log "table rebuilt"
}

seen="$(ls "$LEG"/${STEM}_*k.pth 2>/dev/null | wc -l)"
log "start, leg_200k holds $seen periodic backbones"

while true; do
  now="$(ls "$LEG"/${STEM}_*k.pth 2>/dev/null | wc -l)"
  if [ "$now" -gt "$seen" ]; then
    seen="$now"
    sleep 60                       # let the write finish
    rebuild
  fi
  # the 200k stop, and its GIFT-Eval score, end the watch
  if [ -f "$LEG/${STEM}_200k.pth" ] \
     && [ -f "$STUDY/results/score_k32_bb200k_h30k_student.txt" ]; then
    rebuild
    log "DONE k=32 bb200k scored $(cat "$STUDY/results/score_k32_bb200k_h30k_student.txt")"
    exit 0
  fi
  [ -n "${WATCH_ONCE:-}" ] && { rebuild; exit 0; }
  sleep 600
done
