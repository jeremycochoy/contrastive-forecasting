#!/bin/bash
# Third plateau test — allt·0.8% (the per-domain winner). Chained AFTER the allt·10% follow-up
# so GPU1 is not double-booked. allt·0.8%'s contrastive loss has a long, bumpy temporary plateau:
# its floor-subtracted loss oscillates around 1.3-1.4 over steps ~1000-5000 (bumps up at 1000,
# 2000, 2500, 5000) before descending to ~0.90. Mid-plateau ~= step 2500 (loss 1.41, clearly not
# converged). No intermediary checkpoints were kept, so reproduce step-2500 by re-training with
# the identical config + seed, then score 2L + 6L and compare to the final (2L=1.213, 6L=1.198).
set -uo pipefail
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
SD=$WT/experiments/2026-05-29_forked_6Lf_b1024/scripts
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; LOG="$RES/plateau_followup_allt08.log"; GPU=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# --- wait for the allt·10% follow-up to finish (shared GPU1); 8h backstop ---
log "WAIT for allt·10% follow-up to complete before starting (GPU1 is shared)"
done10=0
for i in $(seq 1 240); do   # 240 * 2min = 8h
  if grep -q "PLATEAU FOLLOWUP (allt·10%) COMPLETE" "$RES/plateau_followup_allt10.log" 2>/dev/null; then
    done10=1; log "allt·10% follow-up complete — starting allt·0.8%"; break
  fi
  sleep 120
done
[ "$done10" = 1 ] || { log "allt·10% did not finish within 8h — aborting to avoid GPU collision"; exit 1; }

# --- 1. reproduce allt·0.8% to step 2500 (mid-plateau); save-every 500 ---
log ">>> re-train allt·0.8% -> step 2500 (mid-plateau)"
QK_NORM=1 ATTN_OUT_NORM=1 LR=1e-3 bash "$SD/train_backbone_b1024_1gpu.sh" \
  alltime 0.0078125 forked2_plat 2500 500 2 "$GPU" >>"$RES/run_allt08_plat_retrain.log" 2>&1
MID="$RUNS/bb_xshh_allt_forked2_plat_6Lf_b1024_FINAL.pth"
log "<<< re-train rc=$? mid_ckpt=$([ -f "$MID" ] && echo OK || echo MISSING)"

# --- 2. fresh 2L + 6L q-heads on the mid-plateau checkpoint + GIFT-Eval ---
if [ -f "$MID" ]; then
  for hl in 2 6; do
    log ">>> allt·0.8% mid-plateau (step2500) ${hl}L downstream"
    bash "$SD/downstream_b1024.sh" "$MID" "$hl" allt08_plat2500 "$GPU" "triage full" \
      >>"$RES/cell_allt08_plat2500_${hl}L.log" 2>&1
    log "<<< allt·0.8% mid-plateau ${hl}L rc=$?"
  done
else
  log "DOWNSTREAM SKIPPED — re-train produced no mid-plateau checkpoint"
fi
log "PLATEAU FOLLOWUP (allt·0.8%) COMPLETE"
