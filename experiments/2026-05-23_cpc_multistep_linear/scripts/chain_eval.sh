#!/bin/bash
# GPU0 chain (#316): all SMALL (2L) heads + the steps-curve. Runs in parallel
# with chain_train.sh (GPU1), which does training + all 6L heads, so each
# backbone's two heads evaluate concurrently.
#   (1) steps-curve triage (small head) on seed A periodic ckpts while FINAL
#       not ready (fills GPU0 during training);
#   (2) seed A FINAL small head triage+full  [headline small, ∥ GPU1 6L];
#   (3) backfill steps-curve;
#   (4) seed B FINAL small head triage+full.
# downstream.sh self-skips evals whose summary.txt exists, so backfill is a no-op
# for points already done.
set -uo pipefail
GPU=0
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
RUNS="$MAIN/runs"
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
A_FINAL="$RUNS/bb_cpc_k12_s20260520_fp32_50k_FINAL.pth"
B_FINAL="$RUNS/bb_cpc_k12_s20260523_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain_eval/G0] $*"; }

for K in 10 20 30 40; do
  [ -f "$A_FINAL" ] && { log "FINAL ready → stop early steps-curve"; break; }
  ck="$RUNS/bb_cpc_k12_s20260520_fp32_50k_${K}k.pth"
  log "steps-curve: waiting for ${K}k (or FINAL)"
  while [ ! -f "$ck" ] && [ ! -f "$A_FINAL" ]; do sleep 60; done
  [ -f "$A_FINAL" ] && { log "FINAL ready → stop early steps-curve"; break; }
  log "steps-curve ${K}k: small head triage"
  bash "$DS" "$ck" 2 "$GPU" triage || true
done

log "waiting for seed A FINAL"
while [ ! -f "$A_FINAL" ]; do sleep 60; done
log "seed A FINAL: small (2L) head → triage+full"
bash "$DS" "$A_FINAL" 2 "$GPU" both || true

for K in 10 20 30 40; do
  ck="$RUNS/bb_cpc_k12_s20260520_fp32_50k_${K}k.pth"
  [ -f "$ck" ] && { log "backfill steps-curve ${K}k (small triage)"; bash "$DS" "$ck" 2 "$GPU" triage || true; }
done

log "waiting for seed B FINAL"
while [ ! -f "$B_FINAL" ]; do sleep 120; done
log "seed B FINAL: small (2L) head → triage+full"
bash "$DS" "$B_FINAL" 2 "$GPU" both || true
log "EVAL CHAIN (GPU0 small heads) COMPLETE"
