#!/bin/bash
# #344 — per-GPU downstream pipeline for one CPC arm: both q-heads (2L, 6L),
# each with best-loss + last checkpoints, then the full-97 evals (DO_EVAL=1).
# Idempotent (skips any cell whose FINAL/summary exists). nohup once per
# arm/GPU after that arm's backbone FINAL exists.
#   chain_cpc.sh <arm: enc3|enc6> <gpu>
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain-cpc-$ARM g$GPU] $*"; }
export DO_EVAL="${DO_EVAL:-1}"
log "chain start (DO_EVAL=$DO_EVAL)"
for HL in 2 6; do
  bash "$SD/downstream_cpc.sh" "$ARM" "$HL" "$GPU" || log "downstream ${HL}L FAILED (continuing)"
done
log "chain complete"
touch "$OUT/results/chain_${ARM}.done"
