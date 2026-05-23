#!/bin/bash
# Eval the k=12 (β-negs) backbone: small (2L) head on GPU0 ∥ 6L head on GPU1,
# each triage+full. Small head is the headline answer vs β (small-head 1.3272).
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
BB="$MAIN/runs/bb_cpcbn_k12_s20260520_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [eval_k12] $*"; }
log "waiting for k=12 FINAL"
while [ ! -f "$BB" ]; do sleep 120; done
log "k=12 FINAL present → small head (GPU0) ∥ 6L head (GPU1)"
( bash "$DS" "$BB" 2 0 both || true ) &
( bash "$DS" "$BB" 6 1 both || true ) &
wait
log "EVAL k=12 COMPLETE"
