#!/bin/bash
# Study arms #2/#3 orchestration. After the two #2 backbones (linbn k=1, k=12)
# finish training, launch #3 (lincn k=1, CPC-negs) on GPU1 and evaluate all
# linear backbones with the SMALL (2L) head (triage+full) on GPU0 for the
# cross-family k-trend table. (#3 k=12 = bb_cpc_k12 already evaluated.)
set -uo pipefail
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
RUN="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/elisa_run_linear.sh"
R="$MAIN/runs"
K12="$R/bb_linbn_k12_s20260520_fp32_50k_FINAL.pth"
K1="$R/bb_linbn_k1_s20260520_fp32_50k_FINAL.pth"
CN1="$R/bb_lincn_k1_s20260520_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain_linear] $*"; }
log "waiting for #2 arms (linbn k=1 + k=12) FINAL"
while [ ! -f "$K12" ] || [ ! -f "$K1" ]; do sleep 120; done
log "#2 done → launch #3 (lincn k=1) on GPU1; eval #2 small head on GPU0"
( bash "$RUN" 20260520 1 1 cpcneg ) &
bash "$DS" "$K12" 2 0 both || true
bash "$DS" "$K1" 2 0 both || true
log "waiting for #3 (lincn k=1) FINAL"
while [ ! -f "$CN1" ]; do sleep 120; done
log "#3 k=1 done → eval small head on GPU0"
bash "$DS" "$CN1" 2 0 both || true
log "LINEAR CHAIN (#2/#3) COMPLETE"
