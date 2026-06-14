#!/bin/bash
# #341 — bulk GIFT-Eval once BOTH arms' heads are done (results/chain_<arm>.done):
# evaluate all cells with LAST checkpoints FIRST (priority), then best-loss,
# pairing the two arms across the two GPUs (arm4/bn on GPU0, arm3/nobn on GPU1),
# 8 shards/cell. Idempotent: run_eval_cell.sh skips any cell whose summary exists
# (so the arm4 2L-last cell already done separately is skipped). Run detached.
set -uo pipefail
SD="$(cd "$(dirname "$0")" && pwd)"
RES=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity/results
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [eval-all] $*"; }
log "waiting for both chains (chain_nobn_enc6.done + chain_bn_enc6.done)"
while [ ! -f "$RES/chain_nobn_enc6.done" ] || [ ! -f "$RES/chain_bn_enc6.done" ]; do sleep 60; done
log "both chains done — both GPUs free; blasting evals, LAST first, 8 shards/cell, bn->GPU0 nobn->GPU1"
pair(){ # <ck: best|last> <head_layers>
  local ck="$1" hl="$2"
  bash "$SD/run_eval_cell.sh" allt08_xftrip_bn_enc6_sgpos_qk_aon_b1024   "$ck" "$hl" 8 0 &
  bash "$SD/run_eval_cell.sh" allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024 "$ck" "$hl" 8 1 &
  wait
}
pair last 2 ; pair last 6     # LAST checkpoints first (highest priority)
pair best 2 ; pair best 6     # then best-loss
log "ALL EVALS DONE"; touch "$RES/all_evals.done"
