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
  echo "phase,k,stop,head_steps,encoder,score"
  for f in "$CF401_RESULTS"/score_k*.txt; do
    [ -e "$f" ] || continue
    [ -s "$f" ] || continue
    score="$(tr -d ' \t\r\n' <"$f")"
    [ -n "$score" ] || continue
    # score_k<K>_bb<label>_h<label>_<enc>.txt, where a label is `40k` or,
    # for a trial budget that is not a multiple of 1000, `400`.
    base="$(basename "$f" .txt)"
    tag="${base#score_}"
    fields="$(printf '%s\n' "$tag" \
      | sed -nE 's/^k([0-9]+)_bb([0-9]+k?)_h([0-9]+k?)_(.+)$/\1 \2 \3 \4/p')"
    [ -n "$fields" ] || { echo "WARN: unparsed score file $base" >&2; continue; }
    read -r k stop_l head_l enc <<<"$fields"
    stop="$(cf401_steps_of "$stop_l")"; head="$(cf401_steps_of "$head_l")"
    phase=1; [ "$head" -eq "$stop" ] && phase=2
    echo "$phase,$k,$stop,$head,$enc,$score"
    split_args+=(--stop "$tag=$(cf401_eval_dir "$k" "$tag")/gift/all_results.csv")
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
