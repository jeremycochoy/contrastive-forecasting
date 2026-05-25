#!/bin/bash
# GPU0 serial queue (#316) — used after the disk-full incident forced a manual
# re-run of seed A's small head. Ensures GPU0 does its work serially:
#   (1) wait for seed A small-head FULL summary (= my manual re-run frees GPU0);
#   (2) wait for seed B FINAL;
#   (3) seed B small (2L) head triage+full on GPU0.
# chain_train.sh handles GPU1: seed B training then seed B 6L head.
set -uo pipefail
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
RUNS="$MAIN/runs"; RES="$MAIN/results"
A_SMALL_FULL="$RES/gift_eval_full_bb_cpc_k12_s20260520_fp32_50k_FINAL_h2L/summary.txt"
B_FINAL="$RUNS/bb_cpc_k12_s20260523_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [gpu0_queue] $*"; }
log "waiting for seed A small-head FULL (GPU0 to free up)"
while [ ! -f "$A_SMALL_FULL" ]; do sleep 60; done
log "seed A small done; waiting for seed B FINAL"
while [ ! -f "$B_FINAL" ]; do sleep 60; done
log "seed B small (2L) head → triage+full on GPU0"
bash "$DS" "$B_FINAL" 2 0 both || true
log "GPU0 QUEUE COMPLETE"
