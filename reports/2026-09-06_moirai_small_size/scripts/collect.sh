#!/bin/bash
# #412 — every number this card produced, in one CSV.
#
# One row for each (arm, stop) pair that has a score. The columns are the
# columns the report groups by, so a figure reads this file and nothing else.
#
#   arm, k, reduce, tau, end, ramp, seed, d_model, params, stop, head_steps,
#   score
#
# `params` is the TRAINABLE count of the shape, from `scripts/model_size.py`.
# It is the number that compares to Moirai-2-Small's 11.4 million, and it is
# not the number a reader gets by summing the checkpoint file.
#
# Usage:  bash scripts/collect.sh [out.csv]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="${1:-$CF412_RESULTS/scores.csv}"
mkdir -p "$(dirname "$OUT")"

PARAMS="$(python3 "$CF412_REPO/scripts/model_size.py" \
  --d-model "$CF412_D_MODEL" --n-heads "$CF412_N_HEADS" \
  --num-layers "$CF412_NUM_LAYERS" \
  --num-encoder-layers "$CF412_NUM_ENCODER_LAYERS" --tsv \
  | awk 'NR == 2 { print $4 }')"

echo "arm,k,reduce,tau,end,ramp,seed,d_model,params,stop,head_steps,score" >"$OUT"
n=0
while read -r arm stop; do
  [ -n "$arm" ] || continue
  score_file="$(cf412_score_file "$arm" "$stop")"
  [ -s "$score_file" ] || continue
  read -r name k red tau end ramp seed <<<"$(cf412_arm_row "$arm")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$arm" "$k" "$red" "$tau" "$end" "$ramp" "$seed" "$CF412_D_MODEL" \
    "$PARAMS" "$stop" "$CF412_HEAD_STEPS" "$(tr -d ' \n' <"$score_file")" \
    >>"$OUT"
  n=$(( n + 1 ))
done < <(cf412_pairs)

echo "wrote $OUT ($n scored pair(s))"
column -s, -t <"$OUT" 2>/dev/null || cat "$OUT"
