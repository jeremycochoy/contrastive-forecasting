#!/bin/bash
# #374 — post-sync orchestrator: wait for sync_loop_374.sh to fetch
# ${NAME}_FINAL.pth from the vast.ai instance, then run downstream on
# elisa's two local GPUs and commit + push produced artefacts.
#
#   orchestrate_after_final_sync.sh
#
# Expects WT, OUT to point at the worktree root and experiment dir.
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
export WT OUT
SCRIPTS="$OUT/scripts"
RES="$OUT/results"; mkdir -p "$RES"
LOG="$RES/orchestrate_after_final_sync.log"
STATE="$RES/orchestrate_state.json"
TAG="split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
BB_FINAL="$OUT/runs/bb_${TAG}_FINAL.pth"
BB_LAST="$OUT/runs/bb_${TAG}_final.pth"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-after-final-sync] $*" | tee -a "$LOG"; }

log "waiting for sync_loop to fetch $BB_FINAL"
while [ ! -f "$BB_FINAL" ]; do sleep 60; done
log "backbone FINAL present locally"
if [ ! -f "$BB_LAST" ]; then
  log "note: last checkpoint not present (${BB_LAST##*/}); downstream last-cell will exit early"
fi

log "downstream launch (2L on GPU 0, 6L on GPU 1)"
GPU_2L="${GPU_2L:-0}" GPU_6L="${GPU_6L:-1}" \
  bash "$SCRIPTS/launch_downstream_split_pred_rep.sh" >>"$LOG" 2>&1
rc_dl=$?
log "downstream done rc=$rc_dl"

log "committing produced artefacts"
cd "$WT"
git add "$OUT/results" "$OUT/README.md" "$OUT/scripts" 2>&1 | tee -a "$LOG"
git add -f "$OUT/results/gift_eval_full_*/all_results.csv" \
  "$OUT/results/gift_eval_full_*/summary.txt" 2>/dev/null || true
if git diff --cached --quiet; then
  log "no new artefacts to commit"
else
  git commit -m "$(cat <<'EOM'
experiment(#374): split_pred_rep downstream artefacts (2L/6L heads + full-97 eval)

Automated commit from orchestrate_after_final_sync.sh: 2L + 6L q-heads on
the best-loss and last-checkpoint split_pred_rep backbones (backbone
trained on vast.ai instance 44361009), full-97 GIFT-Eval B4 summaries
per cell computed on elisa's local 2×RTX 4090.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOM
)" 2>&1 | tee -a "$LOG"
  git push origin "$(git rev-parse --abbrev-ref HEAD)" 2>&1 | tee -a "$LOG"
fi

gm(){ grep 'Aggregate GM-Relative MASE' "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1; }
G_2L_BEST="$(gm "$RES/gift_eval_full_${TAG}_2L/summary.txt")"
G_2L_LAST="$(gm "$RES/gift_eval_full_${TAG}_last_2L/summary.txt")"
G_6L_BEST="$(gm "$RES/gift_eval_full_${TAG}_6L/summary.txt")"
G_6L_LAST="$(gm "$RES/gift_eval_full_${TAG}_last_6L/summary.txt")"
cat > "$STATE" <<EOF
{
  "state": "downstream-done",
  "downstream_rc": $rc_dl,
  "gm_rel_mase": {
    "2L_best":  "${G_2L_BEST:-null}",
    "2L_last":  "${G_2L_LAST:-null}",
    "6L_best":  "${G_6L_BEST:-null}",
    "6L_last":  "${G_6L_LAST:-null}"
  }
}
EOF
log "state written to $STATE"
log "chain complete rc=$rc_dl"
exit "$rc_dl"
