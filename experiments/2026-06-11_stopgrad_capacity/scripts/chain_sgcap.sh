#!/bin/bash
# #341 — per-GPU pipeline: backbone, then all four q-heads for that arm
# (2L best, 2L last, 6L best, 6L last; heads only — evals run sharded via
# shard_evals.py as heads land). Designed to be nohup'd once per arm/GPU at
# launch time; every stage is idempotent (skips work whose FINAL exists).
#   chain_sgcap.sh <arm: nobn_enc6|bn_enc6> <gpu>
set -uo pipefail
ARM="${1:?arm}"; GPU="${2:?gpu}"
SD="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity}"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain-$ARM g$GPU] $*"; }
log "chain start"
bash "$SD/train_backbone_sgcap.sh" "$ARM" "$GPU" || { log "backbone FAILED — stopping chain"; exit 1; }
for HL in 2 6; do
  bash "$SD/downstream_sgcap.sh" "$ARM" "$HL" "$GPU" || log "downstream ${HL}L FAILED (continuing)"
done
log "chain complete"
touch "$OUT/results/chain_${ARM}.done"
