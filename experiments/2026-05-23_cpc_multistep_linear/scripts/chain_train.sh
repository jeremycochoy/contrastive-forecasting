#!/bin/bash
# GPU1 chain (#316): training + all 6L heads. Seed A is launched separately and
# already training on GPU1. This chain then:
#   (1) seed A FINAL 6L head triage+full  [headline 6L, ∥ GPU0 small head];
#   (2) train seed B (20260523) on GPU1;
#   (3) seed B FINAL 6L head triage+full  [∥ GPU0 small head].
# Pairs with chain_eval.sh (GPU0), which does all small heads + the steps-curve.
set -uo pipefail
GPU=1
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
RUN="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/elisa_run.sh"
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
RUNS="$MAIN/runs"
A_FINAL="$RUNS/bb_cpc_k12_s20260520_fp32_50k_FINAL.pth"
B_FINAL="$RUNS/bb_cpc_k12_s20260523_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain_train/G1] $*"; }

log "waiting for seed A FINAL (seed A training already on GPU1)"
while [ ! -f "$A_FINAL" ]; do sleep 120; done
log "seed A FINAL: 6L head → triage+full (GPU1, ∥ GPU0 small head)"
bash "$DS" "$A_FINAL" 6 "$GPU" both || true

log "launching seed B (20260523) training on GPU1"
bash "$RUN" 20260523 1 fp32 || true

log "waiting for seed B FINAL"
while [ ! -f "$B_FINAL" ]; do sleep 60; done
log "seed B FINAL: 6L head → triage+full (GPU1, ∥ GPU0 small head)"
bash "$DS" "$B_FINAL" 6 "$GPU" both || true
log "TRAIN+6L CHAIN (GPU1) COMPLETE"
