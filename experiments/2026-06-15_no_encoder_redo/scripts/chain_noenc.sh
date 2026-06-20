#!/bin/bash
# #348 — per-GPU downstream pipeline for one no-encoder arm: both q-heads
# (2L, 6L), each best-loss + last, then the full-97 evals (DO_EVAL=1).
# Idempotent (skips any cell whose FINAL/summary exists).
#   chain_noenc.sh <arm: base|cpc> <gpu>
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-15_no_encoder_redo}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain-noenc-$ARM g$GPU] $*"; }
export DO_EVAL="${DO_EVAL:-1}"
log "chain start (DO_EVAL=$DO_EVAL)"
for HL in 2 6; do
  bash "$SD/downstream_noenc.sh" "$ARM" "$HL" "$GPU" || log "downstream ${HL}L FAILED (continuing)"
done
log "chain complete"
touch "$OUT/results/chain_${ARM}.done"
