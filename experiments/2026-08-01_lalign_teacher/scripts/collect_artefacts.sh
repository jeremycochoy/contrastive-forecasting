#!/bin/bash
# #390 — assemble the committable artefacts under
# reports/2026-08-01_lalign_teacher/results/, in #379's layout:
#
#   eval_gm_mase/<cell>_summary.txt   one line: aggregate GM-Relative MASE + config count
#   eval_gm_mase/<cell>/all_results.csv   per-config GIFT-Eval output, 97 rows
#   training_curves/<run>_losses.csv  backbone dynamics, DOWNSAMPLED like #379
#   attn_amplitude/<run>_attn_amplitude.csv
#   latent_drift/<run>_latent_drift.csv
#   seasonal_naive_all_results.csv    the denominator of every relative MASE
#   checkpoint_manifest.csv           every checkpoint the run left on elisa
#   logs/                             trainer, eval, orchestrator and pipeline logs
#
# The live `_losses.csv` is ~27 MB per run (one row per step) and there are
# up to 30 of them. #379 committed them downsampled — every step up to 500,
# then every 200th — which is 300 KB and loses nothing the plots read. Same
# rule here, applied by scripts/downsample_curve.py.
#
#   WT=/home/jupyter/wt-cf-390-train REPO=/tmp/contrastive-forecasting-390 \
#     bash collect_artefacts.sh
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"

SRC="$WT/experiments/2026-08-01_lalign_teacher"
# The report is dated by the day the artefacts were collected, not the day
# the runs were launched. Override with DST= to collect somewhere else.
DST="${DST:-$REPO/reports/2026-08-04_lalign_teacher/results}"
mkdir -p "$DST"/{eval_gm_mase,training_curves,attn_amplitude,latent_drift,logs}

say(){ echo "[collect] $*"; }

# --- measurements ----------------------------------------------------------
n_cells=0
for f in "$SRC/eval_gm_mase"/*_summary.txt; do
  [ -e "$f" ] || continue
  cell="$(basename "$f" _summary.txt)"
  csv="$SRC/eval_gm_mase/$cell/gift/all_results.csv"
  rows=0; [ -f "$csv" ] && rows=$(( $(wc -l < "$csv") - 1 ))
  if [ "$rows" -ne 97 ]; then
    say "SKIP $cell — $rows/97 configs, not a complete measurement"
    continue
  fi
  cp -f "$f" "$DST/eval_gm_mase/${cell}_summary.txt"
  mkdir -p "$DST/eval_gm_mase/$cell"
  cp -f "$csv" "$DST/eval_gm_mase/$cell/all_results.csv"
  cp -f "$SRC/eval_gm_mase/$cell/gift/summary.txt" "$DST/eval_gm_mase/$cell/summary.txt" 2>/dev/null
  n_cells=$((n_cells + 1))
done
say "measurements copied: $n_cells complete cells"

# --- training dynamics -----------------------------------------------------
n_curves=0
for f in "$SRC/runs"/*_losses.csv; do
  [ -e "$f" ] || continue
  python3 "$HERE/downsample_curve.py" "$f" "$DST/training_curves/$(basename "$f")" \
    && n_curves=$((n_curves + 1))
done
say "training curves downsampled: $n_curves"

for f in "$SRC/runs"/*_attn_amplitude.csv; do
  [ -e "$f" ] || continue
  cp -f "$f" "$DST/attn_amplitude/$(basename "$f")"
done
for f in "$SRC/runs"/*_latent_drift.csv; do
  [ -e "$f" ] || continue
  cp -f "$f" "$DST/latent_drift/$(basename "$f")"
done

# --- the denominator -------------------------------------------------------
for cand in "$HOME/workspaces/gift-eval/results/seasonal_naive/all_results.csv" \
            "$REPO/reports/2026-07-21_split_pred_rep_small/results/seasonal_naive_all_results.csv"; do
  if [ -f "$cand" ]; then
    cp -f "$cand" "$DST/seasonal_naive_all_results.csv"
    say "seasonal-naive baseline from $cand"
    break
  fi
done

# --- checkpoint manifest ---------------------------------------------------
{
  echo "arm,run_name,checkpoint,step_k,bytes,mtime,sha256"
  for arm in "${CF390_ARMS[@]}"; do
    name="$(bb_name "$arm")" || continue
    for ck in "$SRC/runs/${name}"*.pth; do
      [ -e "$ck" ] || continue
      b="$(basename "$ck")"
      k="$(echo "$b" | grep -oE '_[0-9]+k\.pth$' | tr -dc '0-9')"
      printf '%s,%s,%s,%s,%s,%s,%s\n' "$arm" "$name" "$b" "${k:-}" \
        "$(stat -c%s "$ck")" "$(stat -c%y "$ck" | cut -d. -f1)" \
        "$(sha256sum "$ck" | cut -c1-16)"
    done
  done
} > "$DST/checkpoint_manifest.csv"
say "checkpoint manifest: $(( $(wc -l < "$DST/checkpoint_manifest.csv") - 1 )) checkpoints"

# --- logs ------------------------------------------------------------------
for f in "$SRC/results"/*.log "$SRC/results"/*.tsv "$SRC/results"/*.json \
         "$SRC/results"/*.txt; do
  [ -e "$f" ] || continue
  cp -f "$f" "$DST/logs/$(basename "$f")"
done
for d in "$SRC/eval_gm_mase"/*/; do
  [ -d "$d" ] || continue
  [ -f "$d/eval.log" ] && cp -f "$d/eval.log" "$DST/logs/eval_$(basename "$d").log"
done
say "logs copied: $(ls "$DST/logs" | wc -l) files"
say "done -> $DST"
