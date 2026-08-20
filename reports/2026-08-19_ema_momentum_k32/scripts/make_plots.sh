#!/bin/bash
# #404 — the card's four deliverables, from whatever this study has scored.
#
#   plots/momentum.png         GM-Relative MASE against the EMA momentum
#                              at step 0 (deliverable 1)
#   plots/momentum_at_stop.png the same score against the momentum the arm
#                              HOLDS at the stop. Two ramp lengths now share a
#                              start value, so the step-0 axis puts two arms on
#                              one tick and this axis separates them.
#   plots/loss_curves.png      one training-loss curve per arm, log-log
#                              (deliverable 2)
#   plots/domain_radar.png     GM-Relative MASE per domain (deliverable 3)
#   results/table.md           the table and the statement (deliverable 4)
#   plots/backbone_health.png  the contrastive AUC of every arm against the
#                              backbone step, with any collapsed arm in red
#   results/seed_report.md     one arm at four backbone seeds: which collapsed,
#                              the spread over the rest, and whether that
#                              spread separates 0.90 fixed from 0.95 fixed
#   results/seed_table.csv     the same, as a table a reader does not parse
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

# The health figure, the momentum figure, the table and the seed report read
# the SYNC TREE, not one root: the contrastive AUC lives in each arm's backbone
# losses CSV, and the arms of this card were trained on five boxes.
# `CF404_SYNC_TREE` is the parent of the per-box roots.
SYNC_TREE="${CF404_SYNC_TREE:-$(dirname "$CF404_SYNC_DIR")}"

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
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_PLOTS/momentum.png" \
  --sync-root "$SYNC_TREE"

draw "momentum_at_stop" "$HERE/plot_momentum_at_stop.py" \
  --scores "$CF404_RESULTS/scores.csv" \
  --out "$CF404_PLOTS/momentum_at_stop.png" \
  --sync-root "$SYNC_TREE"

draw "loss_curves" "$HERE/plot_loss_curves.py" \
  --root "$CF404_ROOT" --out "$CF404_PLOTS/loss_curves.png"

draw "domain_radar" "$HERE/plot_domain_radar.py" \
  --splits "$CF404_RESULTS/splits.csv" --out "$CF404_PLOTS/domain_radar.png"

draw "table" "$HERE/make_table.py" \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_RESULTS/table.md" \
  --sync-root "$SYNC_TREE"

draw "backbone_health" "$HERE/plot_backbone_health.py" \
  --sync-root "$SYNC_TREE" --out "$CF404_PLOTS/backbone_health.png"

draw "seed_report" "$HERE/seed_report.py" \
  --scores "$CF404_RESULTS/scores.csv" --sync-root "$SYNC_TREE" \
  --out "$CF404_RESULTS/seed_report.md" \
  --table "$CF404_RESULTS/seed_table.csv"
