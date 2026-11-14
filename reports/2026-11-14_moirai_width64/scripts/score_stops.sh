#!/bin/bash
# #462: the GM-Relative MASE of each stop of the run with its own value heads
# (A2V), the way MPE was scored: strategy A2 with --native-value-head, over
# the 97 GIFT-Eval configs. eval_local.sh of reports/2026-08-08_rollout_depth
# runs the eval in shards on the CPU and reads the score from summary.txt. On
# the 166k checkpoint of MPE, this path gives 6 configs the MASE of the MPE
# table to a relative difference of 6e-6 or less.
#
# Each stop writes
#   results/score_<name>_bb<k>k_native.txt     the GM-Relative MASE
#   results/per_config/<name>_native<k>k.csv   the per-config table, in the
#                                              format of all_results.csv
# and keeps its eval in $R/eval/bb<k>k. A stop is ready when its optimizer
# file exists: the trainer writes it after the weights. A stop with a score
# is skipped. The eval runs the code that the run trains with ($R/code).
#
# Usage: bash score_stops.sh             # score the ready stops, then stop
#        FOLLOW=1 bash score_stops.sh    # then wait for the next stops while
#                                        # train.sh runs ($R/logs/train.pid)
#        DRY_RUN=1 bash score_stops.sh   # print the stops to score
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
R="${CF462_ROOT:-/home/jupyter/cf_runs/cf462}"
RESULTS="${RESULTS:-$STUDY/results}"
CODE="${CODE:-$R/code}"
NAME=cf462_mpe_w64
STOPS="${STOPS:-10 25 50 75 100 125 150 166}"
POLL="${POLL:-600}"

stop_ckpt() {  # the save of stop <k> with its optimizer file
  local c
  for c in "$R"/ckpt/${NAME}_"$1"k.pth "$R"/ckpt/${NAME}_r[0-9]*_"$1"k.pth; do
    if [ -f "${c%.pth}_optimizer.pth" ]; then echo "$c"; return 0; fi
  done
  return 1
}

score_stop() {  # <k> <checkpoint>: the eval, then the table and the score
  local k="$1" ckpt="$2" out="$R/eval/bb$1k"
  local score="$RESULTS/score_${NAME}_bb${k}k_native.txt"
  local table="$RESULTS/per_config/${NAME}_native${k}k.csv"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "score ${k}k $ckpt -> $score $table"; return 0
  fi
  WT="$CODE" EVAL_STRATEGY=A2V EVAL_SHARDS="${EVAL_SHARDS:-8}" \
    CF_BB_SHAPE="--d-model 64 --n-heads 8 --num-layers 3" \
    bash "$CODE/reports/2026-08-08_rollout_depth/scripts/eval_local.sh" \
      "$NAME" "$k" student "$ckpt" none "$out" "$out/score.txt" || return 1
  mkdir -p "$RESULTS/per_config" &&
    cp "$out/gift_a2v/all_results.csv" "$table.tmp" && mv "$table.tmp" "$table" &&
    cp "$out/score.txt" "$score.tmp" && mv "$score.tmp" "$score" || return 1
  echo "$(date -u +%FT%TZ) score: ${k}k = $(cat "$score")"
}

score_ready() {  # one pass. Sets LEFT (stops with no score) and FAILED.
  local k ckpt
  LEFT=0
  for k in $STOPS; do
    [ -s "$RESULTS/score_${NAME}_bb${k}k_native.txt" ] && continue
    if ckpt="$(stop_ckpt "$k")"; then
      score_stop "$k" "$ckpt" || { FAILED=1; LEFT=1; }
    else
      LEFT=1
    fi
  done
}

trainer_runs() {
  [ -f "$R/logs/train.pid" ] && kill -0 "$(cat "$R/logs/train.pid")" 2>/dev/null
}

FAILED=0
score_ready
if [ -n "${FOLLOW:-}" ] && [ -z "${DRY_RUN:-}" ]; then
  while [ "$LEFT" -eq 1 ] && trainer_runs; do
    sleep "$POLL"
    score_ready
  done
  score_ready
fi
exit "$FAILED"
