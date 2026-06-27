#!/bin/bash
# #363 — arm-6 post-experiment finisher.
# Assumes queue_arm6.sh has completed (FINAL.pth + all four gift_eval summary
# .txt files for emb10000_enc10 are on disk).
#
# 1. Run aggregate_to_report.sh (refreshes gm_table.csv, plots, final_traj).
# 2. Stage + commit raw + report artefacts. .pth weights stay ignored.
# 3. Push to feature/contrastive-forecasting-363-v2.
# 4. Post a PR-comment summary on PR #365 opening with
#    «Agent ExperimentRunner claude-opus-4-7 writing».
set -uo pipefail
export WT=/tmp/cf-revert-363
export OUT="$WT/experiments/2026-06-24_sigreg_lambda_sweep"
SUFFIX=emb10000_enc10
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_${SUFFIX}"
RES="$OUT/results"
RUNS="$OUT/runs"
REP="$WT/reports/2026-06-24_sigreg_lambda_sweep"
FLOG="$RES/finish_arm6.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [finish-arm6] $*" | tee -a "$FLOG" >&2; }

cd "$WT"

# Sanity: every cell must be on disk
need=()
for hl in 2L 6L; do
  for kind in '' '_last'; do
    f="$RES/gift_eval_full_${TAG}${kind}_${hl}/summary.txt"
    [ -f "$f" ] || need+=("$f")
  done
done
[ -f "$RUNS/bb_${TAG}_FINAL.pth" ] || need+=("$RUNS/bb_${TAG}_FINAL.pth")
[ -f "$RUNS/bb_${TAG}_losses.csv" ] || need+=("$RUNS/bb_${TAG}_losses.csv")
if [ "${#need[@]}" -gt 0 ]; then
  log "REFUSING to commit — missing: ${need[*]}"
  exit 1
fi

log "running aggregate_to_report.sh"
bash "$REP/scripts/aggregate_to_report.sh" >>"$FLOG" 2>&1
log "aggregate finished (rc=$?)"

log "running compute_bootstrap.py for paired-bootstrap CI refresh"
python3 "$REP/scripts/compute_bootstrap.py" \
  --arm-results "$OUT/results" \
  --sig10-results "$WT/reports/2026-06-22_lejepa_sigreg_emb10/results" \
  --out-dir "$REP" >>"$FLOG" 2>&1
log "compute_bootstrap finished (rc=$?)"

log "running build_plots.py for plot refresh"
python3 "$REP/scripts/build_plots.py" \
  --arm-runs "$REP/runs" \
  --sigreg01-runs "$WT/reports/2026-06-20_lejepa_sigreg/runs" \
  --sigreg10-runs "$WT/reports/2026-06-22_lejepa_sigreg_emb10/runs" \
  --report-dir "$REP" >>"$FLOG" 2>&1
log "build_plots finished (rc=$?)"

# Stage
log "staging artefacts"
git add -A experiments/2026-06-24_sigreg_lambda_sweep/scripts/
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/run_bb_${TAG}.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/sweep_bb_${SUFFIX}.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/sweep_dl_${SUFFIX}.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/dl_2L_${SUFFIX}.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/dl_6L_${SUFFIX}.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/run_qhead_*_${TAG}*.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/run_eval_full_${TAG}*.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/gift_eval_full_${TAG}*/ 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/queue_arm6.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/results/finish_arm6.log 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/runs/bb_${TAG}_losses.csv 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/runs/bb_${TAG}_attn_amplitude.csv 2>/dev/null
git add -A experiments/2026-06-24_sigreg_lambda_sweep/runs/qhead_*_${TAG}*_losses.csv 2>/dev/null
git add -A reports/2026-06-24_sigreg_lambda_sweep/scripts/
git add -A reports/2026-06-24_sigreg_lambda_sweep/runs/
git add -A reports/2026-06-24_sigreg_lambda_sweep/results/
git add -A reports/2026-06-24_sigreg_lambda_sweep/plots/

log "git status:"; git status --short | tee -a "$FLOG" | head -100

# Extract arm-6 GM values for commit message + PR comment
gm2L_best=$(grep -h 'Aggregate GM-Relative MASE' "$RES/gift_eval_full_${TAG}_2L/summary.txt" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
gm2L_last=$(grep -h 'Aggregate GM-Relative MASE' "$RES/gift_eval_full_${TAG}_last_2L/summary.txt" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
gm6L_best=$(grep -h 'Aggregate GM-Relative MASE' "$RES/gift_eval_full_${TAG}_6L/summary.txt" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
gm6L_last=$(grep -h 'Aggregate GM-Relative MASE' "$RES/gift_eval_full_${TAG}_last_6L/summary.txt" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
log "arm-6 GM values: 2L/best=$gm2L_best  2L/last=$gm2L_last  6L/best=$gm6L_best  6L/last=$gm6L_last"

# Commit
COMMITMSG="exp(#363): commit arm-6 raw artefacts (emb10000_enc10, λ_e=1000.0, λ_h=1.0)"
git commit -m "$COMMITMSG" >>"$FLOG" 2>&1
rc=$?
log "git commit rc=$rc"
if [ "$rc" -ne 0 ]; then
  log "git commit failed — see log; not pushing or commenting"
  exit 2
fi

# Push
git push origin feature/contrastive-forecasting-363-v2 >>"$FLOG" 2>&1
rc=$?
log "git push rc=$rc"

# PR comment
BODY=$(cat <<EOF
«Agent ExperimentRunner claude-opus-4-7 writing»

Arm 6 (λ_e=1000.0, λ_h=1.0, suffix \`emb10000_enc10\`) complete. Backbone + downstream + GIFT-Eval full-97 all finished; gm_table.csv, plots, and bootstrap CI artefacts refreshed.

**Experiment directory:** \`experiments/2026-06-24_sigreg_lambda_sweep/\` + \`reports/2026-06-24_sigreg_lambda_sweep/\`

**Runs completed for arm 6 (this comment):**
- backbone train (12 500 steps, GPU 0, ~5 h)
- 2L q-head best + last + GIFT-Eval full-97 (2L on GPU 0)
- 6L q-head best + last + GIFT-Eval full-97 (6L on GPU 1)

**Arm-6 GIFT-Eval full-97 GM-Rel MASE:**

| head | best-ckpt | last-ckpt |
|------|-----------|-----------|
| 2L   | ${gm2L_best} | ${gm2L_last} |
| 6L   | ${gm6L_best} | ${gm6L_last} |

Full updated table: \`reports/2026-06-24_sigreg_lambda_sweep/results/gm_table.csv\` (6 sweep arms + 4 anchors × 4 cells). Paired-bootstrap CIs vs #359 / vs arm 1 refreshed alongside.

Report rewrite to follow from ReportWriter.
EOF
)

gh pr comment 365 --body "$BODY" >>"$FLOG" 2>&1
rc=$?
log "gh pr comment rc=$rc"
log "finisher done"
