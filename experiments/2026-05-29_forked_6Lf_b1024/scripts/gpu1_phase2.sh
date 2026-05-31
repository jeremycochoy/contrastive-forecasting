#!/bin/bash
# #322 phase-2 on the freed GPU1 (after the 5-backbone sweep finished):
#  1. #10 ablation re-train: allt·50% (best arm) → plateau-start step 1000 (~1h).
#  2. MAIN scoreboard: allt·0.8% downstream (2L + 6L) — the last 2 of the 20 cells.
#  3. #10 ablation eval: step-1000 checkpoint (2L + 6L) → compare vs allt·50% final.
# Sequential on GPU1. Idempotent (downstream_b1024.sh skips finished cells; the re-train
# skips if its FINAL exists). GPU0 runs allt·10% downstream in parallel (allt_done launcher).
set -uo pipefail
WT=/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024
SD=$WT/experiments/2026-05-29_forked_6Lf_b1024/scripts
OUT=/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024
RUNS="$OUT/runs"; RES="$OUT/results"; LOG="$RES/gpu1_phase2.log"; GPU=1
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# --- 1. ablation re-train: allt·50% → step 1000 (plateau start) ---
log ">>> #10 ablation re-train: allt·50% -> step1000"
QK_NORM=1 ATTN_OUT_NORM=1 LR=1e-3 bash "$SD/train_backbone_b1024_1gpu.sh" \
  alltime 0.5 forked_step1k 1000 500 2 "$GPU" >>"$RES/run_ablation_retrain.log" 2>&1
ABL="$RUNS/bb_xshh_allt_forked_step1k_6Lf_b1024_FINAL.pth"
log "<<< re-train rc=$? ablation_ckpt=$([ -f "$ABL" ] && echo OK || echo MISSING)"

# --- 2. MAIN: allt·0.8% downstream (the last 2 main-scoreboard cells) ---
for hl in 2 6; do
  log ">>> allt·0.8% ${hl}L (main)"
  bash "$SD/downstream_b1024.sh" "$RUNS/bb_xshh_allt_forked2_qk_aon_6Lf_b1024_FINAL.pth" \
    "$hl" xshh_allt_forked2_qk_aon_b1024 "$GPU" "triage full" \
    >>"$RES/cell_xshh_allt_forked2_qk_aon_b1024_${hl}L.log" 2>&1
  log "<<< allt·0.8% ${hl}L rc=$?"
done

# --- 3. ablation eval: step-1000 checkpoint, 2L + 6L ---
if [ -f "$ABL" ]; then
  for hl in 2 6; do
    log ">>> #10 ablation eval step1k ${hl}L"
    bash "$SD/downstream_b1024.sh" "$ABL" "$hl" allt50_step1k "$GPU" "triage full" \
      >>"$RES/cell_allt50_step1k_${hl}L.log" 2>&1
    log "<<< ablation step1k ${hl}L rc=$?"
  done
else
  log "ABLATION EVAL SKIPPED — re-train produced no checkpoint"
fi
log "GPU1 phase2 COMPLETE"
