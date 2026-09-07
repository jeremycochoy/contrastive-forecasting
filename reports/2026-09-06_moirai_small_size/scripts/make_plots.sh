#!/bin/bash
# #412 — every figure of the report, from the artefacts on disk.
#
# Each script draws what it finds and says how much that was, so this is safe
# to run while the card still runs. Re-run it as each arm lands.
#
# Usage:  bash scripts/make_plots.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
mkdir -p "$CF412_PLOTS"
rc=0
for fig in plot_scores plot_rates plot_climb plot_auc plot_loss_terms; do
  python3 "$HERE/$fig.py" --out "$CF412_PLOTS/${fig#plot_}.png" || rc=1
done
exit $rc
