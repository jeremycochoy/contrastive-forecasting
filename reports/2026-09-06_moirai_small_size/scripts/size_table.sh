#!/bin/bash
# #412 — the parameter count of each candidate shape, and of the model today.
#
# This is the measurement the card's shape decision rests on. It writes two
# files that the report reads:
#
#   results/size_sweep.tsv   one row for each candidate width, at the
#                            published depth of 3 + 3 layers
#   results/size_today.txt   the three counts of the published checkpoint,
#                            which is where the card's 1,135,774 comes from
#
# It needs no GPU and it ends in seconds.
#
# Usage:  bash scripts/size_table.sh [checkpoint]
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/study.sh"

CLI="$CF412_REPO/scripts/model_size.py"
[ -f "$CLI" ] || { echo "ABORT: no size command at $CLI" >&2; exit 2; }
mkdir -p "$CF412_RESULTS"

# The candidates. 256 and 416 bracket the target from both sides, so the table
# shows that 384 is a minimum and not an end point.
WIDTHS="${CF412_WIDTHS:-64,256,320,352,384,416}"
SWEEP="$CF412_RESULTS/size_sweep.tsv"

python3 "$CLI" --d-model-list "$WIDTHS" \
  --num-layers "$CF412_NUM_LAYERS" \
  --num-encoder-layers "$CF412_NUM_ENCODER_LAYERS" \
  --n-heads "$CF412_N_HEADS" --tsv >"$SWEEP" || exit 1
echo "wrote $SWEEP"

python3 "$CLI" --d-model-list "$WIDTHS" \
  --num-layers "$CF412_NUM_LAYERS" \
  --num-encoder-layers "$CF412_NUM_ENCODER_LAYERS" \
  --n-heads "$CF412_N_HEADS" --target "$CF412_TARGET_PARAMS"

# The published checkpoint of the same cell, if this machine holds it. It is
# the source of the card's 1,135,774, and the file tells that number from the
# 720,668 the model trains.
BB="${1:-/home/jupyter/checkpoints_backup/cf-409/dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_200k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_200k.pth}"
if [ -f "$BB" ]; then
  {
    echo "checkpoint: $BB"
    python3 "$CLI" --checkpoint "$BB"
    echo
    echo "the model that wrote it, counted as parameters:"
    python3 "$CLI" --d-model 64 --num-layers 3 --num-encoder-layers 3
  } >"$CF412_RESULTS/size_today.txt" || exit 1
  echo "wrote $CF412_RESULTS/size_today.txt"
else
  echo "NOTE: no published checkpoint at $BB — size_today.txt not written"
fi
