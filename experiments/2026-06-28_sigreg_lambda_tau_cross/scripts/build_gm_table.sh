#!/bin/bash
# #366 — emit reports/gm_table.csv summarising the 8-cell cross
# (arm × head × ckpt → GM-Rel MASE) plus the published anchor cells
# from #363 (SIGReg λ-sweep at τ=0.99) and #357 (SIGReg τ-sweep at
# λ_e=λ_h=0.1) so the cross arms can be read against them.
#
#   bash scripts/build_gm_table.sh
#
# Columns (`source` first so cross/anchor rows separate cleanly):
#   source       — cross | anchor_363 | anchor_357
#   arm,label    — cell tag + readable description
#   head,ckpt    — head depth (2L/6L) × backbone checkpoint (best/last)
#   gm           — GM-Rel MASE = geomean(MASE / SN_MASE), 97 configs
#                  (matches summary.txt's `Aggregate GM-Relative MASE`)
#   gm_mase      — geomean of raw MASE
#   gm_mape_sn   — geomean of MAPE / SN_MAPE
#   gm_crps_sn   — geomean of WQL  / SN_WQL  (WQL = mean_weighted_sum_quantile_loss)
#   n            — number of configs in the geomean (97 expected)
#
# Cross rows are computed from local `results/gift_eval_full_*/all_results.csv`.
# Anchor rows are pulled from the two upstream branches via `git show` (no
# fresh runs); see ARCHIVE_363 / ARCHIVE_357 below for the exact paths.
# Seasonal-Naive reference for the SN-normalised aggregates is loaded from
# ~/workspaces/gift-eval/results/seasonal_naive/all_results.csv by
# scripts/_compute_gm.py.
#
# Idempotent. Writes results/gm_table.csv.
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RES="$EXP_DIR/results"
OUT="$RES/gm_table.csv"
SCRIPT_DIR="$EXP_DIR/scripts"
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc"

# Anchor source-of-truth (branch:path inside that branch).
ARCHIVE_363_BRANCH="origin/feature/contrastive-forecasting-363-v2"
ARCHIVE_363_DIR="experiments/2026-06-24_sigreg_lambda_sweep/results"
ARCHIVE_357_BRANCH="origin/feature/contrastive-forecasting-357"
ARCHIVE_357_DIR="reports/2026-06-21_lejepa_sigreg_tau098/results"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

declare -A ARM_LABEL=(
  [lA_emb100_enc10_tau090]="cross_A: λ_e=10, λ_h=1, τ=0.90"
  [lB_emb10000_enc10_tau090]="cross_B: λ_e=1000, λ_h=1, τ=0.90"
  [lC_emb10_enc10_tau090]="cross_C: λ_e=1, λ_h=1, τ=0.90"
  [lD_emb100_enc100_tau090]="cross_D: λ_e=10, λ_h=10, τ=0.90"
  [lE_emb1000_enc1000_tau090]="cross_E: λ_e=100, λ_h=100, τ=0.90"
  [lF_emb10000_enc10000_tau090]="cross_F: λ_e=1000, λ_h=1000, τ=0.90"
)

# Compute aggregates from a local all_results.csv; echoes a CSV row.
emit_row(){
  local source="$1" arm="$2" label="$3" head="$4" ckpt="$5" csv="$6"
  local vals
  vals=$(python3 "$SCRIPT_DIR/_compute_gm.py" "$csv")
  echo "${source},${arm},\"${label}\",${head},${ckpt},${vals}"
}

# Fetch an anchor cell's all_results.csv from a branch and emit a row.
emit_anchor(){
  local source="$1" branch="$2" anchor_dir="$3" cell="$4" \
        arm="$5" label="$6" head="$7" ckpt="$8"
  local dest="$TMP/${source}_${cell}_${head}_${ckpt}.csv"
  if ! git show "${branch}:${anchor_dir}/${cell}/all_results.csv" \
        >"$dest" 2>/dev/null; then
    echo "MISSING anchor: ${branch}:${anchor_dir}/${cell}/all_results.csv" >&2
    return
  fi
  emit_row "$source" "$arm" "$label" "$head" "$ckpt" "$dest"
}

{
  echo "source,arm,label,head,ckpt,gm,gm_mase,gm_mape_sn,gm_crps_sn,n"

  # --- cross rows (this experiment) ---------------------------------------
  for arm in lA_emb100_enc10_tau090 lB_emb10000_enc10_tau090 \
             lC_emb10_enc10_tau090 lD_emb100_enc100_tau090 \
             lE_emb1000_enc1000_tau090 lF_emb10000_enc10000_tau090; do
    for HL in 2 6; do
      for ckpt in best last; do
        if [ "$ckpt" = best ]; then
          suffix="${TAG}_${arm}_${HL}L"
        else
          suffix="${TAG}_${arm}_last_${HL}L"
        fi
        csv="$RES/gift_eval_full_${suffix}/all_results.csv"
        if [ ! -f "$csv" ]; then
          echo "MISSING: $csv" >&2
          continue
        fi
        emit_row "cross" "cross_${arm:1:1}" "${ARM_LABEL[$arm]}" \
                 "${HL}L" "$ckpt" "$csv"
      done
    done
  done

  # --- anchor rows: #363 (SIGReg λ-sweep, τ=0.99) -------------------------
  # cell tag → (arm-display, label)
  # emb100_enc10   = λ_e=10,   λ_h=1 (the cross_A λ pair, but at τ=0.99)
  # emb10000_enc10 = λ_e=1000, λ_h=1 (the cross_B λ pair, but at τ=0.99)
  for spec in \
      "emb100_enc10|anchor_363_emb100_enc10|#363 λ_e=10, λ_h=1, τ=0.99" \
      "emb10000_enc10|anchor_363_emb10000_enc10|#363 λ_e=1000, λ_h=1, τ=0.99"; do
    IFS='|' read -r cell arm label <<<"$spec"
    for HL in 2 6; do
      for ckpt in best last; do
        if [ "$ckpt" = best ]; then
          cell_dir="${TAG}_${cell}_${HL}L"
        else
          cell_dir="${TAG}_${cell}_last_${HL}L"
        fi
        emit_anchor "anchor_363" "$ARCHIVE_363_BRANCH" "$ARCHIVE_363_DIR" \
                    "gift_eval_full_${cell_dir}" "$arm" "$label" \
                    "${HL}L" "$ckpt"
      done
    done
  done

  # --- anchor rows: #357 (SIGReg τ-sweep, λ_e=λ_h=0.1) --------------------
  # tau090 is the τ=0.90 winner row (matches the cross's τ; λ pair is the
  # #357 reference at λ_e=λ_h=0.1).
  for spec in \
      "tau090|anchor_357_tau090|#357 τ=0.90, λ_e=λ_h=0.1"; do
    IFS='|' read -r cell arm label <<<"$spec"
    for HL in 2 6; do
      for ckpt in best last; do
        if [ "$ckpt" = best ]; then
          cell_dir="${TAG}_${cell}_${HL}L"
        else
          cell_dir="${TAG}_${cell}_last_${HL}L"
        fi
        emit_anchor "anchor_357" "$ARCHIVE_357_BRANCH" "$ARCHIVE_357_DIR" \
                    "gift_eval_full_${cell_dir}" "$arm" "$label" \
                    "${HL}L" "$ckpt"
      done
    done
  done
} >"$OUT"
echo "Wrote $OUT"
cat "$OUT"
