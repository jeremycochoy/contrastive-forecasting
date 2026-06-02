#!/bin/bash
# #322 — EARLY downstream for the two finished β backbones, on GPU0 (user-authorized,
# beside colleagues' rnd_* kernels which sit at ~6.4 GB). 4 cells serial (GPU0 has room
# for one ~10.5 GB q-head at a time, not two): β·0.8% + β·10%, each with a 2L and 6L head.
# Identical #320 q-head recipe + GIFT-Eval (delegates to downstream_b1024.sh). Idempotent:
# downstream_b1024.sh skips a cell whose q-head FINAL + both eval summaries already exist.
set -uo pipefail
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
SD=$WT/experiments/2026-05-29_forked_6Lf_b1024/scripts
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; LOG="$RES/beta_downstream_gpu0.log"
GPU=0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
declare -A BB=(
  [beta_forked2_qk_aon_b1024]=bb_beta_forked2_qk_aon_6Lf_b1024_FINAL.pth
  [beta_forked10pct_qk_aon_b1024]=bb_beta_forked10pct_qk_aon_6Lf_b1024_FINAL.pth
)
# arm-major so each arm (both heads) completes before the next → paired point lands sooner
CELLS=( "beta_forked2_qk_aon_b1024:2" "beta_forked2_qk_aon_b1024:6"
        "beta_forked10pct_qk_aon_b1024:2" "beta_forked10pct_qk_aon_b1024:6" )
log "BETA early downstream START on GPU$GPU — ${#CELLS[@]} cells"
for cell in "${CELLS[@]}"; do
  tag="${cell%%:*}"; hl="${cell##*:}"; bb="$RUNS/${BB[$tag]}"
  [ -f "$bb" ] || { log "SKIP $tag ${hl}L — backbone missing ($bb)"; continue; }
  log ">>> START $tag ${hl}L"
  bash "$SD/downstream_b1024.sh" "$bb" "$hl" "$tag" "$GPU" "triage full" \
    >>"$RES/cell_${tag}_${hl}L.log" 2>&1
  log "<<< DONE $tag ${hl}L rc=$?"
done
log "BETA early downstream COMPLETE"
