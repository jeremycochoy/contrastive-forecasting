#!/bin/bash
# #404 — every score this study wrote, aggregate and per domain.
#
# Two tables, because the card's four deliverables read different things.
#
#   results/scores.csv   one aggregate GM-Relative MASE per arm, with that
#                        arm's momentum beside it. Feeds the momentum figure
#                        and the table.
#   results/splits.csv   the same evals split by horizon term and by domain.
#                        Feeds the radar. The eval publishes no per-domain
#                        number, so it is computed from the eval's own
#                        97-config CSV by #373's `split_scores.py`, against
#                        #379's committed seasonal-naive denominator. One
#                        denominator, or the numbers are not on one scale.
#
# `head_eval_bb.sh` writes one `score_<tag>.txt` per (arm, stop, head budget).
# The tag carries all three, so the table is read back out of the filenames
# rather than kept in a second place that can drift from them.
#
# An empty score file is skipped, not read as 0. An eval killed between
# opening and writing leaves one, and a 0.0 in this CSV would be the best
# GM-Relative MASE the project ever recorded.
#
# Usage:  bash collect.sh          # writes results/scores.csv, splits.csv
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="$CF404_RESULTS/scores.csv"
SPLITS="$CF404_RESULTS/splits.csv"
SPLIT_PY="$CF404_PARENT/scripts/split_scores.py"
mkdir -p "$CF404_RESULTS"

# The `--stop LABEL=CSV` arguments for the per-domain pass, one per scored
# tag. Built in the same loop that reads the aggregates, so the two tables
# always cover the same set of evals.
split_args=()

{
  echo "arm,alpha,schedule,stop,head_steps,encoder,score"
  for f in "$CF404_RESULTS"/score_*.txt; do
    [ -e "$f" ] || continue
    [ -s "$f" ] || continue
    score="$(tr -d ' \t\r\n' <"$f")"
    [ -n "$score" ] || continue
    # score_<arm>_bb<label>_h<label>_<enc>.txt, where a label is `40k` or, for
    # a trial budget that is not a multiple of 1000, `400`.
    base="$(basename "$f" .txt)"
    tag="${base#score_}"
    fields="$(printf '%s\n' "$tag" \
      | sed -nE 's/^(.+)_bb([0-9]+k?)_h([0-9]+k?)_(.+)$/\1 \2 \3 \4/p')"
    [ -n "$fields" ] || { echo "WARN: unparsed score file $base" >&2; continue; }
    read -r arm stop_l head_l enc <<<"$fields"
    # A score file of another study under this results/ is not this card's
    # row. The arms table decides, not the filename shape.
    cf404_is_in "$arm" "$CF404_ARMS" || {
      echo "WARN: $base names '$arm', which is not an arm of this study" >&2
      continue; }
    stop="$(cf404_steps_of "$stop_l")"; head="$(cf404_steps_of "$head_l")"
    echo "$arm,$(cf404_alpha "$arm"),$(cf404_schedule "$arm"),$stop,$head,$enc,$score"
    split_args+=(--stop "$tag=$(cf404_eval_dir "$arm" "$tag")/gift/all_results.csv")
  done
} >"$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"
echo "$OUT: $(( $(wc -l <"$OUT") - 1 )) score(s)"

# The per-domain pass is best effort: it needs the eval's 97-config CSV, which
# a sync loop can still be pulling when this runs. scores.csv is the table the
# figures block on, so a missing split never fails the collect. It says so,
# loudly, instead.
[ -f "$SPLIT_PY" ] || {
  echo "WARN: no split script at $SPLIT_PY — no per-domain table" >&2; exit 0; }
[ "${#split_args[@]}" -gt 0 ] || { echo "$SPLITS: no eval to split yet"; exit 0; }

python3 "$SPLIT_PY" --out "$SPLITS" "${split_args[@]}"
rc=$?
[ $rc -eq 0 ] || echo "WARN: the per-domain pass exited rc=$rc — $SPLITS may be stale" >&2
exit 0
