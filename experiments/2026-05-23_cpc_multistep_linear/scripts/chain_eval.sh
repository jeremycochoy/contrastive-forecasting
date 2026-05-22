#!/bin/bash
# GPU0 eval factory (#316). Priority order:
#   (1) while seed A FINAL not ready: steps-curve triage (small head) on seed A
#       periodic checkpoints as they land (fills the GPU during training);
#   (2) HEADLINE: seed A FINAL — small + 6L head, triage + full;
#   (3) backfill any steps-curve points missed in (1);
#   (4) second seed B FINAL — small + 6L head, triage + full.
# downstream.sh self-skips any eval whose summary.txt already exists, so the
# backfill is a no-op for points already done.
set -uo pipefail
GPU=0
MAIN=/home/jupyter/contrastive-forecasting/experiments/2026-05-23_cpc_multistep_linear
WT=/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound
RUNS="$MAIN/runs"
DS="$WT/experiments/2026-05-23_cpc_multistep_linear/scripts/downstream.sh"
A_FINAL="$RUNS/bb_cpc_k12_s20260520_fp32_50k_FINAL.pth"
B_FINAL="$RUNS/bb_cpc_k12_s20260523_fp32_50k_FINAL.pth"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [chain_eval] $*"; }

# (1) steps-curve points while FINAL not ready (triage, small head)
for K in 10 20 30 40; do
  [ -f "$A_FINAL" ] && { log "FINAL ready → skip remaining early steps-curve"; break; }
  ck="$RUNS/bb_cpc_k12_s20260520_fp32_50k_${K}k.pth"
  log "steps-curve: waiting for ${K}k (or FINAL)"
  while [ ! -f "$ck" ] && [ ! -f "$A_FINAL" ]; do sleep 60; done
  [ -f "$A_FINAL" ] && { log "FINAL ready → skip remaining early steps-curve"; break; }
  log "steps-curve ${K}k: small head triage"
  bash "$DS" "$ck" 2 "$GPU" triage || true
done

# (2) HEADLINE + both-heads: seed A FINAL
log "waiting for seed A FINAL"
while [ ! -f "$A_FINAL" ]; do sleep 60; done
log "seed A FINAL: small (2L) head → triage+full"
bash "$DS" "$A_FINAL" 2 "$GPU" both || true
log "seed A FINAL: 6L head → triage+full"
bash "$DS" "$A_FINAL" 6 "$GPU" both || true

# (3) backfill steps-curve points (triage, small head) for the full curve
for K in 10 20 30 40; do
  ck="$RUNS/bb_cpc_k12_s20260520_fp32_50k_${K}k.pth"
  [ -f "$ck" ] && { log "backfill steps-curve ${K}k"; bash "$DS" "$ck" 2 "$GPU" triage || true; }
done

# (4) second seed B FINAL
log "waiting for seed B FINAL"
while [ ! -f "$B_FINAL" ]; do sleep 120; done
log "seed B FINAL: small (2L) head → triage+full"
bash "$DS" "$B_FINAL" 2 "$GPU" both || true
log "seed B FINAL: 6L head → triage+full"
bash "$DS" "$B_FINAL" 6 "$GPU" both || true
log "EVAL CHAIN COMPLETE"
