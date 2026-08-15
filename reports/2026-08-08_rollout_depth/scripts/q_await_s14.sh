#!/usr/bin/env bash
# #373 session 14 — the await: one line per milestone, and one per failure.
#
# The round has six numbers left: B1's bb200k teacher, A4's bb200k student,
# and the four A1/B3 reproductions. This emits one line when each score file
# lands, one line when a queue job goes to failed, and one line when the
# backbone A4 waits on passes each 5k steps. It exits when all six are in.
set -u
RES="/home/jupyter/wt-cf-373-run2/reports/2026-08-08_rollout_depth/results"
Q="$RES/queue"
A4CSV="/home/jupyter/cf373_r3/sync/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k3_r2_losses.csv"

WANT="B1_k3_bb200k_teacher A4_k3_bb200k_student \
A1rep_k3_bb40k_student B3rep_k3_bb40k_student \
A1rep_k3_bb100k_student B3rep_k3_bb100k_student"

declare -A seen
last_bucket=-1
seen_fail=""

while true; do
  left=0
  for t in $WANT; do
    f="$RES/score_$t.txt"
    if [ -s "$f" ]; then
      if [ -z "${seen[$t]:-}" ]; then
        seen[$t]=1
        echo "SCORE $t = $(tr -d '\n' < "$f")"
      fi
    else
      left=$((left + 1))
    fi
  done

  # any queue job that went to failed, once each
  for s in "$Q"/*.state; do
    [ -f "$s" ] || continue
    st=$(tr -d '\n' < "$s")
    j=$(basename "$s" .state)
    case "$st" in
      failed|error)
        case " $seen_fail " in
          *" $j "*) ;;
          *) seen_fail="$seen_fail $j"; echo "FAILED $j (state=$st)" ;;
        esac
        ;;
    esac
  done

  # A4 backbone progress, one line per 5000 steps
  if [ -s "$A4CSV" ] && [ -z "${seen[A4_k3_bb200k_student]:-}" ]; then
    step=$(tail -1 "$A4CSV" 2>/dev/null | cut -d, -f1)
    case "$step" in
      ''|*[!0-9]*) ;;
      *)
        b=$((step / 5000))
        if [ "$b" -gt "$last_bucket" ]; then
          last_bucket=$b
          echo "A4 backbone step $step / 200000"
        fi
        ;;
    esac
  fi

  [ "$left" -eq 0 ] && { echo "ALL SIX IN"; exit 0; }
  sleep 120
done
