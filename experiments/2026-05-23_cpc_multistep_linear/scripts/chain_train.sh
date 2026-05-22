#!/bin/bash
# GPU1 training chain (#316). Seed A (20260520) is launched separately and
# already running; when its FINAL checkpoint lands, train seed B (20260523)
# on GPU1. Two seeds, identical fp32 recipe → variance estimate.
set -uo pipefail
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
RUN="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/elisa_run.sh"
A="$MAIN/runs/bb_cpc_k12_s20260520_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain_train] $*"; }
log "waiting for seed A FINAL"
while [ ! -f "$A" ]; do sleep 120; done
log "seed A FINAL present → launch seed B (20260523) on GPU1"
bash "$RUN" 20260523 1 fp32
log "TRAIN CHAIN COMPLETE"
