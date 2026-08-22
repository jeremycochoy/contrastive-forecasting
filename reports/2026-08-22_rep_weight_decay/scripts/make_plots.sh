#!/bin/bash
# #409 — every figure of the card, from the artefacts on disk.
#
# Safe to run while the arms train. Each figure draws what is there and says
# how many runs it drew, so a mid-run call gives a mid-run picture.
#
# Usage:  bash scripts/make_plots.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
mkdir -p "$CF409_PLOTS"

bash "$HERE/collect.sh"
python3 "$HERE/plot_auc.py" --root "$CF409_ROOT" --arms "$CF409_ARMS_TSV" \
  --out "$CF409_PLOTS/auc.png"
python3 "$HERE/plot_loss_terms.py" --root "$CF409_ROOT" \
  --arms "$CF409_ARMS_TSV" --out "$CF409_PLOTS/loss_terms.png" \
  --table "$CF409_RESULTS/loss_terms_at_stop.csv"
python3 "$HERE/plot_scores.py" --scores "$CF409_RESULTS/scores.csv" \
  --arms "$CF409_ARMS_TSV" --out "$CF409_PLOTS/scores.png"
