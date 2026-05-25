#!/bin/bash
# Seed B downstream after the resume: 6L head on GPU1 (free once resume finishes)
# in parallel with the small head on GPU0 (once the seed A small-head full eval
# frees GPU0). Both heads → triage+full.
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
R="$MAIN/runs"; RES="$MAIN/results"
B_FINAL="$R/bb_cpc_k12_s20260523_fp32_50k_FINAL.pth"
A_SMALL_FULL="$RES/gift_eval_full_bb_cpc_k12_s20260520_fp32_50k_FINAL_h2L/summary.txt"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [seedB_eval] $*"; }
log "waiting for seed B FINAL (resume to 50k)"
while [ ! -f "$B_FINAL" ]; do sleep 60; done
log "seed B FINAL present → 6L on GPU1 now; small on GPU0 once seed A small frees it"
( bash "$DS" "$B_FINAL" 6 1 both || true ) &
( while [ ! -f "$A_SMALL_FULL" ]; do sleep 60; done; log "GPU0 free → seed B small head"; bash "$DS" "$B_FINAL" 2 0 both || true ) &
wait
log "SEED B EVAL COMPLETE"
