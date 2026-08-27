#!/bin/bash
# #409 — every number the card asks for, in two tables.
#
#   results/scores.csv         one 97-config GM-Relative MASE per arm, KEYED BY
#                              THE EMA SCHEDULE, THE DECAY RAMP AND THE SEED.
#                              Those three values are the arm, so `ema_tau`,
#                              `ema_end`, `ema_ramp`, `ema_at_stop`, `ramp` and
#                              `seed` all ride beside the score. Rows that share
#                              a schedule and a ramp and differ in the seed are
#                              that treatment's spread, not two treatments.
#   results/auc_verdicts.tsv   whether each run held the contrastive task, and
#                              at which step it lost it. The card asks for the
#                              AUC of every run, and a run stopped by the AUC
#                              gate has a verdict and no score.
#
# `head_eval_bb.sh` writes one `score_<tag>.txt` per (arm, stop, head budget).
# The tag carries all three, so the table is read back out of the filenames
# rather than kept in a second place that can drift from them.
#
# An empty score file is skipped, not read as 0. An eval killed between opening
# and writing leaves one, and 0.0 in this CSV would be the best GM-Relative
# MASE the project ever recorded.
#
# Usage:  bash collect.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="$CF409_RESULTS/scores.csv"
VERDICTS="$CF409_RESULTS/auc_verdicts.tsv"
WATCH="$HERE/auc_watch.py"
# Two lanes drain at the same time and the launcher refreshes the table between
# them, so more than one collect can run at once. The temp name carries the
# pid: one shared temp name would let a second collect truncate the file the
# first one is about to move into place.
TMP="$$"
mkdir -p "$CF409_RESULTS"

# ---- The scores --------------------------------------------------------------
#
# `ema_at_stop` is the momentum the arm HOLDS at the stop, which is not the
# momentum its command line names: `dec_m090_r60` and `dec_m090_r200` both name
# 0.9 and hold 0.967 and 0.920. That held value is what ranks two arms.
#
# `ramp` is the decay ramp of the arm, from column 5 of its row.
# `rep_w_at_stop` is the weight the arm HOLDS at the stop. Every arm names 1.0
# at step 0 and every ramp ends well before the stop, so every arm holds 0.0.
{
  echo "arm,ema_tau,ema_end,ema_ramp,ema_at_stop,seed,rep_end,ramp,rep_w_at_stop,align_target,stop,head_steps,encoder,score"
  for f in "$CF409_RESULTS"/score_*.txt; do
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
    # A score file of another study under this results/ is not this card's row.
    # The arms table decides, not the filename shape.
    cf409_is_in "$arm" "$CF409_ARMS" || {
      echo "WARN: $base names '$arm', which is not an arm of this study" >&2
      continue; }
    stop="$(cf409_steps_of "$stop_l")"
    head="$(cf409_steps_of "$head_l")"
    read -r ema_tau ema_end ema_ramp <<<"$(cf409_ema_sig "$arm")"
    echo "$arm,$ema_tau,$ema_end,$ema_ramp,$(cf409_momentum_at "$arm" "$stop"),$(cf409_seed "$arm"),$CF409_REP_W_END,$(cf409_decay_ramp_of "$arm"),$(cf409_rep_w_at "$arm" "$stop"),$CF409_ALIGN_TARGET,$stop,$head,$enc,$score"
  done
} >"$OUT.$TMP"
mv -f "$OUT.$TMP" "$OUT"
echo "$OUT: $(( $(wc -l <"$OUT") - 1 )) score(s)"

# ---- The contrastive AUC -----------------------------------------------------
#
# One row per losses CSV, not per arm: a leg re-fired after a crash resumes
# under a `_rN` run name and writes a second CSV, and the report reads both.
#
# The verdict uses the card's own warmup, which is the gate's. A table built at
# warmup 0 would call the first steps of every healthy run a loss.
csvs=()
for arm in $CF409_ARMS; do
  for stop in $CF409_STOPS; do
    while read -r csv; do
      [ -n "$csv" ] && csvs+=("$csv")
    done < <(cf409_losses_csvs "$arm" "$stop")
  done
done

[ -f "$WATCH" ] || {
  echo "WARN: no watch at $WATCH — no AUC table" >&2; exit 0; }
[ "${#csvs[@]}" -gt 0 ] || { echo "$VERDICTS: no losses CSV yet"; exit 0; }

python3 "$WATCH" "${csvs[@]}" --tsv --window "$CF409_AUC_WINDOW" \
  --threshold "$CF409_AUC_THRESHOLD" --warmup "$CF409_AUC_WARMUP" \
  >"$VERDICTS.$TMP"
rc=$?
mv -f "$VERDICTS.$TMP" "$VERDICTS"
# Exit code 1 means a run lost the task, which is a RESULT of this card and not
# a failure of the collect. Only a broken CSV (2) is worth a warning.
[ "$rc" -le 1 ] || echo "WARN: the AUC pass exited rc=$rc — see $VERDICTS" >&2
echo "$VERDICTS: $(( $(wc -l <"$VERDICTS") - 1 )) run(s)"
exit 0
