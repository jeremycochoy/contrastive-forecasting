#!/bin/bash
# #401 — every score this study wrote, aggregate and per domain.
#
# Two tables, because the card asks for two deliverables and they read
# different things.
#
#   results/scores.csv   one aggregate GM-Relative MASE per (depth, stop,
#                        head budget). Feeds the ladder figure and the
#                        phase-2 picker.
#   results/splits.csv   the same evals split by horizon term and by domain.
#                        Feeds the radar figure. The eval publishes no
#                        per-domain number, so it is computed from the eval's
#                        own 97-config CSV by #373's `split_scores.py`,
#                        against #379's committed seasonal-naive denominator.
#                        One denominator, or the numbers are not on one scale.
#
# `head_eval_bb.sh` writes one `score_<tag>.txt` per (depth, stop, head
# budget). The tag carries all three, so the table is read back out of the
# filenames rather than kept in a second place that can drift from them.
#
# A tag can carry a fourth part: a VARIANT, between the depth and the stop.
# `k32_ema30k_bb40k_h30k_student` is the card's cell at k = 32 with the EMA
# ramp shortened to 30,000 steps. It is a different training schedule at the
# same (depth, stop, head budget), so it is a different cell, and the
# `variant` column is what says so. Without that column the variant and the
# base cell share one key, and a reader — or a plot — takes the second one
# read as the first one's score.
#
# `base` is the card's own schedule. Every grid cell is `base`.
#
# The phase is derived, not stored: a head budget equal to the backbone stop
# is phase 2, anything else is phase 1. That is the card's own definition of
# the two phases.
#
# An empty score file is skipped, not read as 0. An eval killed between
# opening and writing leaves one, and a 0.0 in this CSV would be the best
# GM-Relative MASE the project ever recorded.
#
# Usage:  bash collect.sh            # writes results/scores.csv, splits.csv
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="$CF401_RESULTS/scores.csv"
SPLITS="$CF401_RESULTS/splits.csv"
SPLIT_PY="$CF401_PARENT/scripts/split_scores.py"
mkdir -p "$CF401_RESULTS"

# The `--stop LABEL=CSV` arguments for the per-domain pass, one per scored
# tag. Built in the same loop that reads the aggregates, so the two tables
# always cover the same set of evals.
split_args=()

{
  echo "phase,k,variant,stop,head_steps,encoder,score"
  for f in "$CF401_RESULTS"/score_k*.txt; do
    [ -e "$f" ] || continue
    [ -s "$f" ] || continue
    score="$(tr -d ' \t\r\n' <"$f")"
    [ -n "$score" ] || continue
    # score_k<K>[_<variant>]_bb<label>_h<label>_<enc>.txt, where a label is
    # `40k` or, for a trial budget that is not a multiple of 1000, `400`.
    #
    # Two anchored patterns, not one with an optional group. An optional
    # group that matches nothing prints an empty field, `read` then collapses
    # the two spaces around it, and every field after it shifts left by one —
    # so a base cell would take its stop as its variant and lose its encoder.
    stem="$(basename "$f" .txt)"
    tag="${stem#score_}"
    fields="$(printf '%s\n' "$tag" \
      | sed -nE 's/^k([0-9]+)_bb([0-9]+k?)_h([0-9]+k?)_(.+)$/\1 base \2 \3 \4/p')"
    [ -n "$fields" ] || fields="$(printf '%s\n' "$tag" \
      | sed -nE 's/^k([0-9]+)_([a-z][a-z0-9]*)_bb([0-9]+k?)_h([0-9]+k?)_(.+)$/\1 \2 \3 \4 \5/p')"
    [ -n "$fields" ] || { echo "WARN: unparsed score file $stem" >&2; continue; }
    read -r k variant stop_l head_l enc <<<"$fields"
    stop="$(cf401_steps_of "$stop_l")"; head="$(cf401_steps_of "$head_l")"
    phase=1; [ "$head" -eq "$stop" ] && phase=2
    echo "$phase,$k,$variant,$stop,$head,$enc,$score"
    # `cf401_eval_csv` falls back to a path under the default root when it
    # resolves nothing, and `split_scores.py` skips a file that is not there.
    # Together those two turn a cell with no reachable eval into a cell that
    # is simply absent from splits.csv, and the per-domain figure then draws
    # one panel fewer with no line anywhere saying which cell it lost. This
    # is that line.
    eval_csv="$(cf401_eval_csv "$k" "$tag" "$variant")"
    if [ -f "$eval_csv" ]; then
      split_args+=(--stop "$tag=$eval_csv")
    else
      echo "WARN: $tag has a score but no per-domain CSV at $eval_csv" >&2
      echo "  it will be absent from $SPLITS and from the per-domain figure" >&2
    fi
  done
} >"$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"
echo "$OUT: $(( $(wc -l <"$OUT") - 1 )) score(s)"

# The per-domain pass is best effort: it needs the eval's 97-config CSV,
# which a sync loop can still be pulling when this runs. scores.csv is the
# table phase 2 blocks on, so a missing split never fails the collect. It
# says so, loudly, instead.
[ -f "$SPLIT_PY" ] || {
  echo "WARN: no split script at $SPLIT_PY — no per-domain table" >&2; exit 0; }
[ "${#split_args[@]}" -gt 0 ] || { echo "$SPLITS: no eval to split yet"; exit 0; }

python3 "$SPLIT_PY" --out "$SPLITS" "${split_args[@]}"
rc=$?
[ $rc -eq 0 ] || echo "WARN: the per-domain pass exited rc=$rc — $SPLITS may be stale" >&2
exit 0
