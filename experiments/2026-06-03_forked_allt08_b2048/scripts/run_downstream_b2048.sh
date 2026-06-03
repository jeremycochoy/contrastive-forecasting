#!/bin/bash
# #327 — downstream driver. Once the b2048 backbone FINAL exists, train a 2L and a 6L
# q-head and run GIFT-Eval triage-11 + full-97 for each. The two (head) cells run in
# parallel, one per card: 2L on GPU1, 6L on GPU0 (q-heads are ~10.5 GB so the 6L cell
# fits on GPU0 beside the foreign tenants). Per-cell script is idempotent (skips finished
# q-heads/evals), so re-running resumes. Waits for the backbone FINAL before starting.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_forked_allt08_b2048
RUNS="$OUT/runs"; RES="$OUT/results"; mkdir -p "$RES"
DLOG="$RES/downstream.log"
BB="$RUNS/bb_xshh_allt_forked2_qk_aon_6Lf_b2048_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [dl] $*" | tee -a "$DLOG"; }

while [ ! -f "$BB" ]; do log "waiting for backbone FINAL $(basename "$BB")"; sleep 120; done
log "backbone FINAL present — launching 2L (GPU1) + 6L (GPU0) in parallel"
bash "$HERE/downstream_b2048.sh" "$BB" 2 forked2_qk_aon 1 "triage full" >>"$RES/cell_2L.log" 2>&1 &
P2=$!
bash "$HERE/downstream_b2048.sh" "$BB" 6 forked2_qk_aon 0 "triage full" >>"$RES/cell_6L.log" 2>&1 &
P6=$!
wait $P2; r2=$?
wait $P6; r6=$?
log "DOWNSTREAM DONE (2L rc=$r2, 6L rc=$r6)"
touch "$RES/.downstream_done"
