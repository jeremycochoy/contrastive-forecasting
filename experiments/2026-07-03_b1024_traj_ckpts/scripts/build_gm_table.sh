#!/bin/bash
# #369 — emit results/gm_table.csv comparing the B=1024 retrain against
# the parent B=512 arm from #366's τ=0.90 grid.
#
# Rows:
#   retrain, ${SUFFIX} @ parent-step  → head × ckpt: 2L/best, 2L/last, 6L/best, 6L/last
#   retrain, ${SUFFIX} @ last (12,500) → same 4 cells
#   parent  , #366 τ=0.90 cell           → same 4 cells (fetched from #366's branch)
#
# The retrain rows come from local `results/gift_eval_full_*/all_results.csv`.
# The parent row is pulled from the #366 branch via `git show` (no fresh
# runs). ARCHIVE_366 defaults below can be overridden on the command line.
#
#   bash scripts/build_gm_table.sh
#   ARCHIVE_366_BRANCH=... ARCHIVE_366_CELL=... bash scripts/build_gm_table.sh
#
# Idempotent. Writes results/gm_table.csv.
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RES="$EXP_DIR/results"
OUT="$RES/gm_table.csv"
SCRIPT_DIR="$EXP_DIR/scripts"
WINNERS_FILE="${WINNERS_FILE:-$EXP_DIR/winners.sh}"
[ -f "$WINNERS_FILE" ] || { echo "ABORT: winners manifest not found at $WINNERS_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
. "$WINNERS_FILE"

# Suffix derivation mirrors launch_experiment.sh's suffix_for().
SUFFIX=$(awk -v le="$LAMBDA_E" -v lh="$LAMBDA_H" -v t="$TAU" \
  'BEGIN { printf "l_emb%.0f_enc%.0f_tau%03.0f_b1024\n", le*10, lh*10, t*100 }')
TAG="allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_${SUFFIX}"
LABEL_RETRAIN="retrain: λ_e=${LAMBDA_E}, λ_h=${LAMBDA_H}, τ=${TAU}, B=1024"

# Parent's cell tag on #366's grid. `emb${10·λ_e}_enc${10·λ_h}_tau${100·τ}` —
# same encoding as launch_experiment.sh but without the `_b1024` marker.
# Override at the command line if the parent lives under a different cell
# label (e.g. arm C → `lC_emb10_enc10_tau090`).
# Parent aggregates come from #366's committed gm_table.csv (arm C's raw
# per-task CSVs were never committed — only the 4-cell aggregates). Pin a
# commit rather than a branch so drift can't change the numbers.
ARCHIVE_366_COMMIT="${ARCHIVE_366_COMMIT:-ba1df52}"
ARCHIVE_366_GM_TABLE="${ARCHIVE_366_GM_TABLE:-experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv}"
PARENT_ARM="${PARENT_ARM:-cross_C}"

# Extra trajectory steps scored by dl_at_step.sh (space-separated, may be
# empty). Each has a `gift_eval_full_${TAG}_step<N>_{2L,6L}` results dir.
EXTENDED_STEPS="${EXTENDED_STEPS:-15000 20000 25000 30000 35000 37500 40000 45000 50000}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

emit_row(){
  local source="$1" arm="$2" label="$3" head="$4" ckpt="$5" csv="$6"
  local vals
  if ! vals=$(python3 "$SCRIPT_DIR/_compute_gm.py" "$csv" 2>/dev/null); then
    echo "SKIP (unreadable per-task CSV): $csv" >&2
    return 0
  fi
  echo "${source},${arm},\"${label}\",${head},${ckpt},${vals}"
}

emit_from_local(){
  local source="$1" arm="$2" label="$3" head="$4" ckpt="$5" cell="$6"
  local csv="$RES/gift_eval_full_${cell}_${head}/all_results.csv"
  if [ ! -f "$csv" ]; then
    echo "MISSING: $csv" >&2
    return
  fi
  emit_row "$source" "$arm" "$label" "$head" "$ckpt" "$csv"
}

{
  echo "source,arm,label,head,ckpt,gm,gm_mase,gm_mape_sn,gm_crps_sn,n"

  # Retrain rows: two loci × 2L/6L
  # Cell 1: retrained backbone at parent's best-loss step
  for HL in 2L 6L; do
    emit_from_local "retrain" "b1024_parentstep" "${LABEL_RETRAIN} @ parent-best-step (${PARENT_BEST_LOSS_STEP})" \
                    "$HL" "step${PARENT_BEST_LOSS_STEP}" "${TAG}_parentstep"
  done
  # Cell 2: retrained backbone at last (12,500)
  for HL in 2L 6L; do
    emit_from_local "retrain" "b1024_last" "${LABEL_RETRAIN} @ last (12,500)" \
                    "$HL" "step12500" "${TAG}_last"
  done
  # Extended-trajectory rows (follow-up sweep past the issue's 12,500).
  for STEP in $EXTENDED_STEPS; do
    for HL in 2L 6L; do
      emit_from_local "retrain" "b1024_step${STEP}" "${LABEL_RETRAIN} @ step ${STEP}" \
                      "$HL" "step${STEP}" "${TAG}_step${STEP}"
    done
  done

  # Parent rows: copied verbatim from #366's committed gm_table.csv
  # (aggregates only; raw per-task CSVs were never committed).
  if ! git -C "$EXP_DIR" show "${ARCHIVE_366_COMMIT}:${ARCHIVE_366_GM_TABLE}" 2>/dev/null \
      | awk -F, -v arm="$PARENT_ARM" '$2==arm {print "parent_366," $0}' \
      | sed 's/^parent_366,cross,/parent_366,/'; then
    echo "MISSING parent rows: ${ARCHIVE_366_COMMIT}:${ARCHIVE_366_GM_TABLE} arm=${PARENT_ARM}" >&2
  fi
} >"$OUT"
echo "Wrote $OUT"
cat "$OUT"
