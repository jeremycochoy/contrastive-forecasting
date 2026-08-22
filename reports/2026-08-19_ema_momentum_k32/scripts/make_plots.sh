#!/bin/bash
# #404 — the card's four deliverables, from whatever this study has scored.
#
#   plots/arm_ranking.png      every run, one row per arm, ordered by score
#   plots/reached_vertical.png the score against the momentum the arm HOLDS at
#                              the stop (deliverable 1)
#   plots/by_start.png         the score against the start of the schedule
#   plots/by_ramp.png          the score against the length of the ramp
#   plots/loss_terms.png       the total loss and each term that makes it
#                              (deliverable 2)
#   plots/domain_grid.png      GM-Relative MASE per domain (deliverable 3)
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
# The root this study READS on elisa is the sync tree, the same one
# `evals_elisa.sh` takes. Without this line CF404_ROOT falls back to the
# durable local root, which holds no arm of this card: every arm trained on a
# box. `plot_loss_curves.py --root` then found no curve and the figure was
# SKIPPED with a one-line message, so a stale loss_curves.png survived a
# redraw. `CF404_ROOT_GIVEN` keeps an operator's own root winning.
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_SYNC_ROOT"

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

# Every figure the report embeds is drawn here. The four score figures were
# drawn by hand for one round, and a redraw then missed them.
draw "arm_ranking" "$HERE/plot_arm_ranking.py" \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_PLOTS/arm_ranking.png" \
  --sync-root "$SYNC_TREE"

draw "reached_vertical" "$HERE/plot_reached_two_colours.py" --vertical \
  --scores "$CF404_RESULTS/scores.csv" \
  --out "$CF404_PLOTS/reached_vertical.png" --sync-root "$SYNC_TREE"

draw "by_start" "$HERE/plot_two_axes.py" --by start \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_PLOTS/by_start.png" \
  --sync-root "$SYNC_TREE"

draw "by_ramp" "$HERE/plot_two_axes.py" --by ramp \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_PLOTS/by_ramp.png" \
  --sync-root "$SYNC_TREE"

# The loss decomposition reads the COMMITTED curves, so a redraw with no sync
# tree still draws it. `s08b` is the run whose contrastive AUC fell to 0.57.
LT_BASE="$HERE/../curves/box_a/cf393_arm6_v2_combab_alignT_cf373k32_mean"
LT_ARGS=()
for tag in a08 a095 a09 r100_08 r100_095 r100_09b r100_09 r60_09 \
           s08c s08d s08 s09 w3_s08; do
  LT_ARGS+=(--curve "13 runs that held=${LT_BASE}_${tag}_losses.csv")
done
LT_ARGS+=(--curve "!1 whose contrastive AUC fell to 0.57=${LT_BASE}_s08b_losses.csv")
draw "loss_terms" "$HERE/plot_loss_terms.py" "${LT_ARGS[@]}" \
  --grey-red --cols 3 --turn 500 --out "$CF404_PLOTS/loss_terms.png"

draw "domain_grid" "$HERE/plot_domain_grid.py" \
  --splits "$CF404_RESULTS/splits.csv" --out "$CF404_PLOTS/domain_grid.png"

draw "table" "$HERE/make_table.py" \
  --scores "$CF404_RESULTS/scores.csv" --out "$CF404_RESULTS/table.md" \
  --sync-root "$SYNC_TREE"

draw "backbone_health" "$HERE/plot_backbone_health.py" \
  --sync-root "$SYNC_TREE" --out "$CF404_PLOTS/backbone_health.png"

draw "seed_report" "$HERE/seed_report.py" \
  --scores "$CF404_RESULTS/scores.csv" --sync-root "$SYNC_TREE" \
  --out "$CF404_RESULTS/seed_report.md" \
  --table "$CF404_RESULTS/seed_table.csv"
