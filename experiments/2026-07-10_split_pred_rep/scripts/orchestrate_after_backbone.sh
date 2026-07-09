#!/bin/bash
# #374 — chain the split_pred_rep experiment end-to-end AFTER the backbone
# training PID is done. Waits for the backbone process, then runs downstream
# (2L + 6L in parallel on the two local GPUs), then commits every produced
# artefact (logs, CSVs, checkpoints omitted per per-class .gitignore), then
# pushes.
#
#   orchestrate_after_backbone.sh <backbone-pid>
#
# Expects WT, OUT to point at the worktree root and experiment dir.
set -uo pipefail
BB_PID="${1:?backbone pid}"
: "${WT:?WT must be set}"
: "${OUT:?OUT must be set}"
export WT OUT
SCRIPTS="$OUT/scripts"
RES="$OUT/results"; mkdir -p "$RES"
LOG="$RES/orchestrate_after_backbone.log"
STATE="$RES/orchestrate_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-after-bb] $*" | tee -a "$LOG"; }

log "waiting for backbone PID $BB_PID"
while kill -0 "$BB_PID" 2>/dev/null; do sleep 60; done
log "backbone PID $BB_PID no longer running"

TAG="split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"
BB_FINAL="$OUT/runs/bb_${TAG}_FINAL.pth"
BB_LAST="$OUT/runs/bb_${TAG}_final.pth"
if [ ! -f "$BB_FINAL" ]; then
  log "ABORT: backbone FINAL missing at $BB_FINAL — training did not complete"
  echo "{\"state\":\"backbone-failed\",\"final\":\"$BB_FINAL\"}" > "$STATE"
  exit 1
fi
if [ ! -f "$BB_LAST" ]; then
  log "ABORT: last checkpoint missing at $BB_LAST — training did not emit _final.pth"
  echo "{\"state\":\"backbone-partial\",\"final\":\"$BB_FINAL\",\"last\":\"$BB_LAST\"}" > "$STATE"
  exit 1
fi
log "backbone artefacts present: FINAL + last"

log "downstream launch (2L + 6L parallel)"
GPU_2L="${GPU_2L:-0}" GPU_6L="${GPU_6L:-1}" \
  bash "$SCRIPTS/launch_downstream_split_pred_rep.sh" >>"$LOG" 2>&1
rc_dl=$?
log "downstream done rc=$rc_dl"

log "committing produced artefacts (logs, CSVs, eval summaries; not .pth)"
cd "$WT"
git add "$OUT/results" "$OUT/README.md" "$OUT/scripts" 2>&1 | tee -a "$LOG"
git add -f "$OUT/results/gift_eval_full_*/all_results.csv" \
  "$OUT/results/gift_eval_full_*/summary.txt" 2>/dev/null || true
if git diff --cached --quiet; then
  log "no new artefacts to commit"
else
  git commit -m "$(cat <<'EOM'
experiment(#374): split_pred_rep downstream artefacts (backbone-final + heads + full-97)

Automated commit from orchestrate_after_backbone.sh: 2L + 6L q-heads on
the best-loss and last-checkpoint split_pred_rep backbones, full-97
GIFT-Eval B4 summaries per cell.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOM
)" 2>&1 | tee -a "$LOG"
  git push origin "$(git rev-parse --abbrev-ref HEAD)" 2>&1 | tee -a "$LOG"
fi

# Emit a state sentinel with headline GM values so the orchestrator turn
# can read them off disk without re-running gh calls.
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
