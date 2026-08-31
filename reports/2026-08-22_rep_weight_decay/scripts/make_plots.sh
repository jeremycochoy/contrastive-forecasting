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
# The verdict table says which run lost the contrastive task, and the two
# curve figures paint that run in the alarm colour. `collect.sh` above wrote it.
VERDICTS="$CF409_RESULTS/auc_verdicts.tsv"
# `plot_auc.py` shades the decay ramp. It reads the ramp of each arm from the
# arms table, so it takes no ramp here.
python3 "$HERE/plot_auc.py" --root "$CF409_ROOT" --arms "$CF409_ARMS_TSV" \
  --verdicts "$VERDICTS" --out "$CF409_PLOTS/auc.png"
python3 "$HERE/plot_loss_terms.py" --root "$CF409_ROOT" \
  --arms "$CF409_ARMS_TSV" --verdicts "$VERDICTS" \
  --out "$CF409_PLOTS/loss_terms.png" \
  --table "$CF409_RESULTS/loss_terms_at_stop.csv"
python3 "$HERE/plot_scores.py" --scores "$CF409_RESULTS/scores.csv" \
  --arms "$CF409_ARMS_TSV" --out "$CF409_PLOTS/scores.png"
# The score on each axis: the momentum at the stop, and the decay ramp.
python3 "$HERE/plot_axes.py" --scores "$CF409_RESULTS/scores.csv" \
  --arms "$CF409_ARMS_TSV" --out "$CF409_PLOTS/axes.png"
# The measured grid: every (decay ramp, momentum) cell the card scored.
python3 "$HERE/plot_grid.py" --scores "$CF409_RESULTS/scores.csv" \
  --arms "$CF409_ARMS_TSV" --verdicts "$VERDICTS" --out "$CF409_PLOTS/grid.png"
# The slope at the stop, which is the card's second question.
python3 "$HERE/loss_slope.py" --root "$CF409_ROOT" --arms "$CF409_ARMS_TSV" \
  --out "$CF409_RESULTS/loss_slope.csv"
# What this card can rank, and whether its reference is comparable. This card
# runs no control, so both are part of reading any number it produced.
python3 "$HERE/rank_gate.py" --scores "$CF409_RESULTS/scores.csv" \
  --arms "$CF409_ARMS_TSV" --out "$CF409_RESULTS/rank_gate.tsv"
bash "$HERE/reference_match.sh" "$CF409_RESULTS/reference_match.tsv"
