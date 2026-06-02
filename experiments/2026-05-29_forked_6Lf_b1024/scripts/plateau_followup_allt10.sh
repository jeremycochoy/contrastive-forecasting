#!/bin/bash
# #322 plateau follow-up (PR #324 review, 2026-06-01): re-run the plateau ablation on the
# BEST SCORER (allt·10%) to test whether the "training past the plateau does not help"
# trend seen on allt·50% (#10) generalises.
#
# The temporary plateau: allt·10%'s floor-subtracted contrastive loss STALLS / bumps up
# across steps ~1500->3000 (1.19 -> 1.20 -> 1.22) before resuming its descent to ~0.85.
# Mid-plateau ~= step 2500. No intermediary checkpoints were kept for allt·10%, so we
# reproduce the step-2500 state by re-training with the IDENTICAL config + seed
# (deterministic), then score it with fresh 2L and 6L q-heads and compare to the final.
set -uo pipefail
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
SD=$WT/experiments/2026-05-29_forked_6Lf_b1024/scripts
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; LOG="$RES/plateau_followup_allt10.log"; GPU=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# 1. reproduce allt·10% to step 2500 (mid-plateau); save-every 500 keeps 500..2500
log ">>> re-train allt·10% (best scorer) -> step 2500 (mid-plateau), save-every 500"
QK_NORM=1 ATTN_OUT_NORM=1 LR=1e-3 bash "$SD/train_backbone_b1024_1gpu.sh" \
  alltime 0.10 forked10pct_plat 2500 500 2 "$GPU" >>"$RES/run_allt10_plat_retrain.log" 2>&1
MID="$RUNS/bb_xshh_allt_forked10pct_plat_6Lf_b1024_FINAL.pth"
log "<<< re-train rc=$? mid_ckpt=$([ -f "$MID" ] && echo OK || echo MISSING)"

# 2. fresh 2L + 6L q-heads on the mid-plateau checkpoint + GIFT-Eval (triage + full)
if [ -f "$MID" ]; then
  for hl in 2 6; do
    log ">>> allt·10% mid-plateau (step2500) ${hl}L downstream"
    bash "$SD/downstream_b1024.sh" "$MID" "$hl" allt10_plat2500 "$GPU" "triage full" \
      >>"$RES/cell_allt10_plat2500_${hl}L.log" 2>&1
    log "<<< allt·10% mid-plateau ${hl}L rc=$?"
  done
else
  log "DOWNSTREAM SKIPPED — re-train produced no mid-plateau checkpoint"
fi
log "PLATEAU FOLLOWUP (allt·10%) COMPLETE"
