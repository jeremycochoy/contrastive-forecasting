#!/bin/bash
# #401 — the card's two deliverables, from whatever this study has scored.
#
#   plots/depth_ladder.png          GM-Relative MASE against backbone train
#                                   step, both phases (deliverable 2).
#   plots/domain_radar_phase1.png   per-domain, one panel per depth
#   plots/domain_radar_phase2.png   (deliverable 1).
#   plots/arm_compare.png           this arm beside the summed arm, which is
#   results/arm_compare.csv / .md   the pair this protocol exists to draw.
#
# It runs `collect.sh` first, so both tables are current, then draws from
# them. A figure with no input is skipped with a line saying so, never with a
# stack trace: phase 2 has no rows until phase 1 has picked its arms, and
# this is meant to run at any point in the study.
#
# The comparison reads BOTH arms, so its two paths are absolute and not
# `$CF401_RESULTS`: the summed arm's table is `results/scores.csv` and this
# arm's is `$CF401_RESULTS/scores.csv`. Either may hold fewer cells than the
# other, or none.
#
# Usage:  bash scripts/make_plots.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

PLOTS="$CF401_PLOTS"
mkdir -p "$PLOTS"

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

draw "depth_ladder" "$HERE/plot_depth_ladder.py" \
  --scores "$CF401_RESULTS/scores.csv" --out "$PLOTS/depth_ladder.png"

for phase in 1 2; do
  draw "domain_radar_phase$phase" "$HERE/plot_domain_radar.py" \
    --splits "$CF401_RESULTS/splits.csv" --phase "$phase" \
    --out "$PLOTS/domain_radar_phase${phase}.png"
done

# The comparison. `SUM_SCORES` is the stopped arm's table, which lives at the
# study's own results directory whatever this arm's is.
SUM_SCORES="$CF401_STUDY/results/scores.csv"
draw "arm_compare table" "$HERE/compare_arms.py" \
  --sum "$SUM_SCORES" --mean "$CF401_RESULTS/scores.csv" \
  --out "$CF401_RESULTS/arm_compare.csv"
draw "arm_compare" "$HERE/plot_arm_compare.py" \
  --sum "$SUM_SCORES" --mean "$CF401_RESULTS/scores.csv" \
  --out "$PLOTS/arm_compare.png"
