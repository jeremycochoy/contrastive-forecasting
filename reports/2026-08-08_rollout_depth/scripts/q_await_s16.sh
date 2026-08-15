#!/usr/bin/env bash
# #373 session 16 — the await for the last five numbers.
#
# Left in the round: A4's bb200k student, and the four A1/B3 student
# reproductions. This emits one line when each score lands, one line when a
# queue job goes to failed, one line per 5,000 steps of A4's backbone, and
# one line per hour with the credit. It exits when all five are in, or after
# MAXH hours.
set -u
RES="/home/jupyter/wt-cf-373-run2/reports/2026-08-08_rollout_depth/results"
Q="$RES/queue"
A4CSV="/home/jupyter/cf373_r3/sync/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k3_r2_losses.csv"
MAXH="${1:-8}"

WANT="A4_k3_bb200k_student \
A1rep_k3_bb40k_student B3rep_k3_bb40k_student \
A1rep_k3_bb100k_student B3rep_k3_bb100k_student"

declare -A seen
last_bucket=-1
seen_fail=""
t0=$(date +%s)
last_hb=0

while true; do
  left=0
  for t in $WANT; do
    f="$RES/score_$t.txt"
    if [ -s "$f" ]; then
      if [ -z "${seen[$t]:-}" ]; then
        seen[$t]=1
        echo "[$(date -u '+%H:%M:%SZ')] SCORE $t = $(tr -d '\n' < "$f")"
      fi
    else
      left=$((left + 1))
    fi
  done

  for s in "$Q"/*.state; do
    [ -f "$s" ] || continue
    st=$(tr -d '\n' < "$s")
    j=$(basename "$s" .state)
    case "$st" in
      failed|error)
        case " $seen_fail " in
          *" $j "*) ;;
          *) seen_fail="$seen_fail $j"
             echo "[$(date -u '+%H:%M:%SZ')] FAILED $j (state=$st)" ;;
        esac
        ;;
    esac
  done

  # A4 backbone progress, one line per 5,000 steps.
  if [ -s "$A4CSV" ]; then
    step=$(tail -1 "$A4CSV" | cut -d, -f1)
    case "$step" in
      ''|*[!0-9]*) ;;
      *)
        b=$(( step / 5000 ))
        if [ "$b" -ne "$last_bucket" ]; then
          last_bucket=$b
          echo "[$(date -u '+%H:%M:%SZ')] A4 backbone step $step / 200000"
        fi ;;
    esac
  fi

  now=$(date +%s)
  if [ $(( now - last_hb )) -ge 3600 ]; then
    last_hb=$now
    cred=$(timeout 60 vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')
    echo "[$(date -u '+%H:%M:%SZ')] HEARTBEAT left=$left credit=${cred:-?}"
  fi

  [ "$left" -eq 0 ] && { echo "[$(date -u '+%H:%M:%SZ')] ALL FIVE IN"; exit 0; }
  if [ $(( now - t0 )) -ge $(( MAXH * 3600 )) ]; then
    echo "[$(date -u '+%H:%M:%SZ')] TIMEOUT after ${MAXH}h, left=$left"; exit 2
  fi
  sleep 120
done
