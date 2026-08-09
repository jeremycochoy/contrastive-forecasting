#!/bin/bash
# #373 — the horizon splits, the paired bootstraps and the tables for the
# runs the experiment review asked for.
#
# Idempotent and re-runnable: it re-derives everything from the per-config
# CSVs on disk, so running it while the queue is still going gives the
# tables for whatever has finished. bootstrap_gaps.csv is rewritten each
# time rather than appended to, so a re-run never doubles a row.
#
# Usage: bash gap_analyse.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${RES:-$STUDY/results}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
EVAL="$CF373_ROOT/eval"
mkdir -p "$RES"

csv_of(){ printf '%s/%s/gift/all_results.csv\n' "$EVAL" "$1"; }
have(){ [ -s "$(csv_of "$1")" ]; }

# ---- horizon and domain splits, every stop that has a per-config CSV ------
stops=()
for d in "$EVAL"/*/; do
  tag="$(basename "$d")"
  have "$tag" && stops+=("--stop" "$tag=$(csv_of "$tag")")
done
if [ "${#stops[@]}" -gt 0 ]; then
  python3 "$HERE/split_scores.py" --out "$RES/splits_all.csv" "${stops[@]}" \
    >"$RES/splits_all.log" 2>&1
  echo "splits: $(( ${#stops[@]} / 2 )) stop(s) -> $RES/splits_all.csv"
fi

# ---- paired bootstraps ----------------------------------------------------
# Each row is (label, the arm that is the baseline, the arm compared to it).
# The seed rows are the point of gap 5: same recipe, same code, same head
# seed, same eval, different backbone seed — the spread no run of this study
# or its parents had measured.
OUT="$RES/bootstrap_gaps.csv"
rm -f "$OUT"
pairs=(
  "B9_depth|G2_B9_k0|B9_k3"
  "A3_alignx4|A3_k0|G3_A3_k0_aw4"
  "A3_depth1|A3_k0|G3_A3_k1"
  "B5_seed2_depth|G5_B5_s2_k0|G5_B5_s2_k3"
  "B5_backboneseed_k0|B5_k0|G5_B5_s2_k0"
  "B5_backboneseed_k3|B5_k3|G5_B5_s2_k3"
  "B1_depth|G6_B1_k0|G6_B1_k3"
)
for spec in "${pairs[@]}"; do
  IFS='|' read -r label a b <<<"$spec"
  for enc in student teacher; do
    ta="${a}_bb40k_${enc}"; tb="${b}_bb40k_${enc}"
    have "$ta" && have "$tb" || continue
    python3 "$HERE/paired_bootstrap.py" \
      --k0 "$(csv_of "$ta")" --k3 "$(csv_of "$tb")" \
      --label "${label}_${enc}" --out "$OUT"
  done
done
[ -s "$OUT" ] && echo "bootstrap -> $OUT"

# ---- the tables -----------------------------------------------------------
python3 "$HERE/gap_table.py" --results "$RES" --out "$RES/gap_scores.md"
