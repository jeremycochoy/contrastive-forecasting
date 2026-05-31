#!/bin/bash
# #322 — early downstream for the two FINISHED all-time arms (allt·50% = xshh_allt_forked,
# allt·10% = xshh_allt_forked10pct) on GPU0, overlapping arm 5's training on GPU1. 4 cells
# serial (one ~10.5GB q-head fits GPU0 beside the colleagues' kernels). Identical #320 recipe
# (downstream_b1024.sh). Idempotent. The allt·0.8% arm (arm 5) cells run later once its FINAL
# lands (both GPUs free then).
set -uo pipefail
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
SD=$WT/experiments/2026-05-29_forked_6Lf_b1024/scripts
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; LOG="$RES/allt_done_downstream_gpu0.log"
GPU=0
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
declare -A BB=(
  [xshh_allt_forked_qk_aon_b1024]=bb_xshh_allt_forked_qk_aon_6Lf_b1024_FINAL.pth
  [xshh_allt_forked10pct_qk_aon_b1024]=bb_xshh_allt_forked10pct_qk_aon_6Lf_b1024_FINAL.pth
)
CELLS=( "xshh_allt_forked_qk_aon_b1024:2" "xshh_allt_forked_qk_aon_b1024:6"
        "xshh_allt_forked10pct_qk_aon_b1024:2" "xshh_allt_forked10pct_qk_aon_b1024:6" )
log "ALLT-done early downstream START on GPU$GPU — ${#CELLS[@]} cells"
for cell in "${CELLS[@]}"; do
  tag="${cell%%:*}"; hl="${cell##*:}"; bb="$RUNS/${BB[$tag]}"
  [ -f "$bb" ] || { log "SKIP $tag ${hl}L — backbone missing ($bb)"; continue; }
  log ">>> START $tag ${hl}L"
  bash "$SD/downstream_b1024.sh" "$bb" "$hl" "$tag" "$GPU" "triage full" \
    >>"$RES/cell_${tag}_${hl}L.log" 2>&1
  log "<<< DONE $tag ${hl}L rc=$?"
done
log "ALLT-done early downstream COMPLETE"
