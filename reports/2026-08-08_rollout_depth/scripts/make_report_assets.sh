#!/bin/bash
# #373 — rebuild every figure and table from whatever results exist.
#
# Usage: bash make_report_assets.sh [git checkout root]
#
# Idempotent and partial-tolerant: a cell that has not finished is skipped
# with a line saying so, rather than taking the whole rebuild down. Run it
# again when the cell lands.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_ROOT="${1:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
RES="$DST/results"; PLOTS="$DST/plots"
SYNC_BASE="${CF373_SYNC_BASE:-$HOME/cf373_sync}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
BATCH="$GIT_ROOT/reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt"
mkdir -p "$RES" "$PLOTS"
export PYTHONPATH="${PYTHONPATH:-}:$GIT_ROOT"
say(){ echo "[assets] $*"; }

# ---- 1. splits: horizon term and domain, per finished stop -----------------
stops=()
for d in "$RES"/eval/*/; do
  [ -f "$d/all_results.csv" ] || continue
  n=$(( $(wc -l <"$d/all_results.csv") - 1 ))
  if [ "$n" -ne 97 ]; then
    say "skip $(basename "$d"): $n configs, want 97"; continue
  fi
  stops+=(--stop "$(basename "$d")=$d/all_results.csv")
done
if [ "${#stops[@]}" -gt 0 ]; then
  python3 "$HERE/split_scores.py" "${stops[@]}" --out "$RES/splits.csv" \
    | sed 's/^/  /'
else
  say "no finished eval yet — no splits, no score figures"
fi

# ---- 2. score figures ------------------------------------------------------
if [ -f "$RES/splits.csv" ]; then
  for head in student teacher; do
    python3 "$HERE/plot_horizon_split.py" --splits "$RES/splits.csv" \
      --head "$head" --out "$PLOTS/horizon_split_${head}.png" 2>&1 \
      | grep -v Warning | sed 's/^/  /'
    python3 "$HERE/plot_domain_radar.py" --splits "$RES/splits.csv" \
      --head "$head" --out "$PLOTS/domain_radar_${head}.png" 2>&1 \
      | grep -v Warning | sed 's/^/  /'
  done
  python3 "$HERE/plot_k3_vs_k0.py" --splits "$RES/splits.csv" \
    --out "$PLOTS/k3_vs_k0.png" 2>&1 | grep -v Warning | sed 's/^/  /'
fi

# ---- 3. training curves ----------------------------------------------------
# One losses CSV per (cell, k), found by the run name cell_paths.sh builds.
. "$HERE/cell_paths.sh"
runs=()
while IFS=$'\t' read -r id slug launcher arg; do
  case "$id" in ''|'#'*) continue ;; esac
  for k in 0 3; do
    name="$(cf373_run_name "$id" "$k")" || continue
    csv="$(find "$SYNC_BASE" -type f -name "${name}_losses.csv" 2>/dev/null | head -1)"
    [ -n "$csv" ] && [ "$(wc -l <"$csv")" -gt 10 ] && runs+=(--run "$id:$k=$csv")
  done
done < "$HERE/cells.tsv"

if [ "${#runs[@]}" -gt 0 ]; then
  python3 "$HERE/plot_cos_err_depth.py" "${runs[@]}" \
    --out "$PLOTS/cos_err_depth.png" 2>&1 | grep -v Warning | sed 's/^/  /'
  python3 "$HERE/plot_train_curves.py" "${runs[@]}" --out-dir "$PLOTS" 2>&1 \
    | grep -v Warning | sed 's/^/  /'
else
  say "no losses CSV synced yet — no training-curve figures"
fi

# ---- 4. rollout fidelity ---------------------------------------------------
fid=()
while IFS=$'\t' read -r id slug launcher arg; do
  case "$id" in ''|'#'*) continue ;; esac
  for k in 0 3; do
    bb="$(cf373_bb_ckpt "$id" "$k" 40000)" || continue
    [ -n "$bb" ] && fid+=(--run "${id}_k${k}=$bb")
  done
done < "$HERE/cells.tsv"

if [ "${#fid[@]}" -gt 0 ] && [ -f "$BATCH" ]; then
  python3 "$HERE/rollout_fidelity.py" "${fid[@]}" --batch "$BATCH" \
    --out "$RES/rollout_fidelity.csv" 2>&1 | grep -v Warning | sed 's/^/  /'
  [ -f "$RES/rollout_fidelity.csv" ] && \
    python3 "$HERE/plot_rollout_fidelity.py" --csv "$RES/rollout_fidelity.csv" \
      --out "$PLOTS/rollout_fidelity.png" 2>&1 | grep -v Warning | sed 's/^/  /'
else
  say "no bb40k checkpoint on the durable root yet — no fidelity figure"
fi

# ---- 5. the score table ----------------------------------------------------
python3 "$HERE/score_table.py" --results "$RES" --out "$RES/scores.md" 2>&1 \
  | sed 's/^/  /'

say "-> $PLOTS"
ls -1 "$PLOTS" 2>/dev/null | sed 's/^/  /'
