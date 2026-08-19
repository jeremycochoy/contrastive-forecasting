#!/bin/bash
# #404 — the card's four deliverables, from whatever this study has scored.
#
#   plots/momentum.png      GM-Relative MASE against the EMA momentum
#                           (deliverable 1)
#   plots/loss_curves.png   one training-loss curve per arm, log-log
#                           (deliverable 2)
#   plots/domain_radar.png  GM-Relative MASE per domain (deliverable 3)
#   results/table.md        the table and the statement (deliverable 4)
#
# It runs `collect.sh` first, so both tables are current, then draws from them.
# A figure with no input is SKIPPED with a line saying so, never with a stack
# trace: this runs every 30 minutes while the study climbs, and the radar has
# no rows until the first eval finishes.
#
# Usage:  bash scripts/make_plots.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

mkdir -p "$CF404_PLOTS" "$CF404_RESULTS"

bash "$HERE/collect.sh" || exit $?

draw(){  # <name> <script> <args...>
  local name="$1"; shift
  local out rc
  out="$(python3 "$@" 2>&1)"; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "$out"
  else
    echo "SKIP $name (rc=$rc): $(printf '%s' "$out" | tail -1)"
  fi
}

draw "momentum" "$HERE/plot_momentum.py" \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_PLOTS/momentum.png"

draw "loss_curves" "$HERE/plot_loss_curves.py" \
  --root "$CF404_ROOT" --out "$CF404_PLOTS/loss_curves.png"

draw "domain_radar" "$HERE/plot_domain_radar.py" \
  --splits "$CF404_RESULTS/splits.csv" --out "$CF404_PLOTS/domain_radar.png"

draw "table" "$HERE/make_table.py" \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_RESULTS/table.md"
