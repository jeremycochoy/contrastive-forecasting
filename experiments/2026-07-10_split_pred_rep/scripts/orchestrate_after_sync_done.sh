#!/bin/bash
# #374 — final orchestrator: waits for all 4 downstream summary.txt files
# to arrive on elisa (via the sync loop), then commits and pushes.
#
#   orchestrate_after_sync_done.sh
set -uo pipefail
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
RES="$OUT/results"; mkdir -p "$RES"
LOG="$RES/orchestrate_after_sync_done.log"
STATE="$RES/orchestrate_state.json"
TAG=split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090
CELLS=("${TAG}_2L" "${TAG}_last_2L" "${TAG}_6L" "${TAG}_last_6L")

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-sync-done] $*" | tee -a "$LOG"; }
all_done(){
  for c in "${CELLS[@]}"; do
    [ -f "$OUT/results/gift_eval_full_$c/summary.txt" ] || return 1
  done
  return 0
}

log "waiting for all 4 summary.txt files"
while ! all_done; do sleep 60; done
log "all 4 cells present locally"

cd "$WT"
git add "$OUT/results" "$OUT/README.md" "$OUT/scripts" 2>&1 | tee -a "$LOG"
git add -f "$OUT/results/gift_eval_full_"*"/all_results.csv" \
  "$OUT/results/gift_eval_full_"*"/summary.txt" 2>/dev/null || true
if git diff --cached --quiet; then
  log "no new artefacts to commit"
else
  git commit -m "$(cat <<'EOM'
experiment(#374): split_pred_rep downstream artefacts (2L/6L × best/last)

Automated commit: q-head trained on both split_pred_rep backbone
checkpoints (best_loss + last) at head_layers ∈ {2, 6}, full-97
GIFT-Eval B4 per (head, ckpt) cell. Backbone trained on vast.ai
(#376 c03aeb1/4edc212c); downstream started on elisa (best-loss
q-heads + partial evals) then migrated to a second vast.ai instance
to free elisa GPU 0 / 1 for other experiments.

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
  "gm_rel_mase": {
    "2L_best":  "${G_2L_BEST:-null}",
    "2L_last":  "${G_2L_LAST:-null}",
    "6L_best":  "${G_6L_BEST:-null}",
    "6L_last":  "${G_6L_LAST:-null}"
  }
}
EOF
log "state written; downstream complete"
