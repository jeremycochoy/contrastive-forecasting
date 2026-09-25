#!/bin/bash
# #412 and #414 — the table of scores and every figure of the report, from the
# artefacts on disk. Safe to run while an arm still trains: each script draws
# what it finds.
#
# Usage:  bash scripts/make_plots.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
for step in gm_trajectories plot_gm_rates plot_414 plot_hard plot_radar_value; do
  python3 "$HERE/$step.py" || rc=1
done
exit $rc
