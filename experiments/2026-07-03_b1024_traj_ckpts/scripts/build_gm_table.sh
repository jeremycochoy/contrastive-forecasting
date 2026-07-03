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
PARENT_CELL=$(awk -v le="$LAMBDA_E" -v lh="$LAMBDA_H" -v t="$TAU" \
  'BEGIN { printf "emb%.0f_enc%.0f_tau%03.0f\n", le*10, lh*10, t*100 }')
PARENT_TAG_PREFIX="${PARENT_TAG_PREFIX:-allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc}"
# `lX_` prefix — set from the actual winner arm letter at launch time
# (see README §Arms). Defaults to `l_` (no letter); override in env.
PARENT_ARM_LETTER="${PARENT_ARM_LETTER:-}"
PARENT_CELL_FULL="${PARENT_ARM_LETTER}${PARENT_CELL}"
ARCHIVE_366_BRANCH="${ARCHIVE_366_BRANCH:-origin/feature/contrastive-forecasting-366-title-tweak}"
ARCHIVE_366_DIR="${ARCHIVE_366_DIR:-experiments/2026-06-28_sigreg_lambda_tau_cross/results}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

emit_row(){
  local source="$1" arm="$2" label="$3" head="$4" ckpt="$5" csv="$6"
  local vals
  vals=$(python3 "$SCRIPT_DIR/_compute_gm.py" "$csv")
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

emit_from_366(){
  local source="$1" arm="$2" label="$3" head="$4" ckpt="$5" cell="$6"
  local dest="$TMP/${source}_${cell}_${head}_${ckpt}.csv"
  if ! git show "${ARCHIVE_366_BRANCH}:${ARCHIVE_366_DIR}/gift_eval_full_${cell}_${head}/all_results.csv" \
        >"$dest" 2>/dev/null; then
    echo "MISSING parent: ${ARCHIVE_366_BRANCH}:${ARCHIVE_366_DIR}/gift_eval_full_${cell}_${head}/all_results.csv" >&2
    return
  fi
  emit_row "$source" "$arm" "$label" "$head" "$ckpt" "$dest"
}

{
  echo "source,arm,label,head,ckpt,gm,gm_mase,gm_mape_sn,gm_crps_sn,n"

  # Retrain rows: two loci × 2L/6L
  # Cell 1: retrained backbone at parent's best-loss step
  for HL in 2L 6L; do
    emit_from_local "retrain" "b1024_parentstep" "${LABEL_RETRAIN} @ parent-best-step (${PARENT_BEST_LOSS_STEP})" \
                    "$HL" "best" "${TAG}_parentstep"
  done
  # Cell 2: retrained backbone at last (12,500)
  for HL in 2L 6L; do
    emit_from_local "retrain" "b1024_last" "${LABEL_RETRAIN} @ last (12,500)" \
                    "$HL" "last" "${TAG}_last"
  done

  # Parent rows: fetched from #366's branch. Cover all four cells so the
  # comparison shows best-ckpt AND last-ckpt sides.
  parent_label="parent (#366 τ=${TAU}, λ_e=${LAMBDA_E}, λ_h=${LAMBDA_H}, B=512)"
  for HL in 2L 6L; do
    for ckpt in best last; do
      if [ "$ckpt" = best ]; then
        cell="${PARENT_TAG_PREFIX}_${PARENT_CELL_FULL}"
      else
        cell="${PARENT_TAG_PREFIX}_${PARENT_CELL_FULL}_last"
      fi
      emit_from_366 "parent_366" "parent_366_${PARENT_CELL_FULL}" "$parent_label" \
                    "$HL" "$ckpt" "$cell"
    done
  done
} >"$OUT"
echo "Wrote $OUT"
cat "$OUT"
