#!/bin/bash
# #412 — the three deliverables of the card, in three tables.
#
#   results/scores.csv         the 97-config GM-Relative MASE of each
#                              (arm, stop). The arm's depth, reduction,
#                              momentum, seed and decay ride beside it.
#   results/auc_verdicts.tsv   whether each run held the contrastive task, and
#                              at which step it lost it.
#   results/loss_terms.csv     the training loss by term at each stop:
#                              L_pred, L_rep and L_align, with the live L_rep
#                              weight and the live EMA momentum.
#
# The card asks for all three. `L_rep` holds 92 to 93 percent of the total
# loss on this cell, so the total alone says almost nothing about the other
# two terms, and a decay arm's whole treatment is the weight on that term.
#
# `params` is the TRAINABLE count of the shape, from `scripts/model_size.py`.
# It is the number that compares to Moirai-2-Small's 11.4 million, and it is
# not the number a reader gets by summing the checkpoint file.
#
# An empty score file is skipped, not read as 0. An eval killed between
# opening and writing leaves one, and 0.0 would be the best GM-Relative MASE
# the project ever recorded.
#
# Usage:  bash scripts/collect.sh [out.csv]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

OUT="${1:-$CF412_RESULTS/scores.csv}"
TERMS="$CF412_RESULTS/loss_terms.csv"
VERDICTS="$CF412_RESULTS/auc_verdicts.tsv"
mkdir -p "$(dirname "$OUT")" "$CF412_RESULTS"

PARAMS="$(python3 "$CF412_REPO/scripts/model_size.py" \
  --d-model "$CF412_D_MODEL" --n-heads "$CF412_N_HEADS" \
  --num-layers "$CF412_NUM_LAYERS" \
  --num-encoder-layers "$CF412_NUM_ENCODER_LAYERS" --tsv \
  | awk 'NR == 2 { print $4 }')"

# ---- The scores --------------------------------------------------------------
echo "arm,k,reduce,tau,end,ramp,seed,decay,d_model,params,stop,head_steps,score" \
  >"$OUT"
n=0
while read -r arm stop; do
  [ -n "$arm" ] || continue
  score_file="$(cf412_score_file "$arm" "$stop")"
  [ -s "$score_file" ] || continue
  read -r name k red tau end ramp seed decay <<<"$(cf412_arm_row "$arm")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$arm" "$k" "$red" "$tau" "$end" "$ramp" "$seed" "$decay" \
    "$CF412_D_MODEL" "$PARAMS" "$stop" "$CF412_HEAD_STEPS" \
    "$(tr -d ' \n' <"$score_file")" >>"$OUT"
  n=$(( n + 1 ))
done < <(cf412_pairs)
echo "wrote $OUT ($n scored pair(s))"

# ---- The training loss by term -----------------------------------------------
#
# One row for each (arm, stop) that has a losses CSV, read at the LAST row at
# or below the stop. #409 added `l_pred`, `l_rep`, `l_align` and `rep_w` to
# the trainer's CSV, so this reads them by NAME: the column order of that file
# depends on the rollout depth, which differs by arm.
#
# The whole curve stays in the losses CSV under the checkpoint root. This
# table is the point a report reads at each stop.
echo "arm,stop,step,loss,l_pred,l_rep,l_align,rep_w,ema_tau,auc" >"$TERMS"
m=0
while read -r arm stop; do
  [ -n "$arm" ] || continue
  row=""
  while read -r csv; do
    [ -n "$csv" ] || continue
    found="$(awk -F',' -v stop="$stop" '
      NR == 1 { for (i = 1; i <= NF; i++) { gsub(/\r/, "", $i); col[$i] = i }
                next }
      col["step"] && $col["step"] + 0 <= stop + 0 {
        out = $col["step"]
        n = split("loss l_pred l_rep l_align rep_w ema_tau auc", want, " ")
        for (i = 1; i <= n; i++) {
          v = col[want[i]] ? $col[want[i]] : ""
          gsub(/\r/, "", v)
          out = out "," v
        }
        last = out
      }
      END { if (last != "") print last }' "$csv")"
    [ -n "$found" ] && row="$found"
  done < <(cf412_losses_csvs "$arm" "$stop")
  [ -n "$row" ] || continue
  printf '%s,%s,%s\n' "$arm" "$stop" "$row" >>"$TERMS"
  m=$(( m + 1 ))
done < <(cf412_pairs)
echo "wrote $TERMS ($m stop(s) with a loss row)"

# ---- The contrastive AUC -----------------------------------------------------
#
# One row per losses CSV, not per arm: a leg re-fired after a crash resumes
# under a `_rN` run name and writes a second CSV, and the report reads both.
#
# The verdict uses the card's own warm-up, which is the gate's. A table built
# at warm-up 0 would call the first steps of every healthy run a loss.
csvs=()
for arm in $CF412_ARMS; do
  for stop in $CF412_STOPS; do
    while read -r csv; do
      [ -n "$csv" ] && csvs+=("$csv")
    done < <(cf412_losses_csvs "$arm" "$stop")
  done
done

if [ -f "$CF412_AUC_WATCH_PY" ] && [ "${#csvs[@]}" -gt 0 ]; then
  python3 "$CF412_AUC_WATCH_PY" "${csvs[@]}" --tsv \
    --window "$CF412_AUC_WINDOW" --threshold "$CF412_AUC_THRESHOLD" \
    --warmup "$CF412_AUC_WARMUP" >"$VERDICTS"
  rc=$?
  # 1 means a run lost the task, which is a RESULT of this card and not a
  # failure of the collect. Only a broken CSV (2) is worth a warning.
  [ "$rc" -le 1 ] || echo "WARN: the AUC pass exited rc=$rc — see $VERDICTS" >&2
  echo "wrote $VERDICTS ($(( $(wc -l <"$VERDICTS") - 1 )) run(s))"
  # A lost run is the loudest line of this collect, so it repeats on stdout.
  awk -F'\t' '$2 == "lost" { print "LOST: " $1 " at step " $3 }' "$VERDICTS"
else
  echo "$VERDICTS: no losses CSV yet"
fi

column -s, -t <"$OUT" 2>/dev/null || cat "$OUT"
exit 0
