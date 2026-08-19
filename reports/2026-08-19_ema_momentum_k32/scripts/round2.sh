#!/bin/bash
# #404 round 2 — the three added arms, from bare boxes to three scores.
#
# The review of PR #405 asks for three arms on top of the card's four:
#
#   a085  the fixed value the card names itself
#   a095  the fixed value above 0.9, where the data points
#   s08b  s08 again at a second backbone seed, which measures the repeat
#         spread of THIS cell
#
# Each arm takes its own single-card box (`round2_box.sh`), and the three run
# at the same time. This driver then does what needs every arm:
#
#   1. the three boxes, in parallel
#   2. the three 97-config GIFT-Evals, here, on the CPU
#   3. the shard check, the artefacts and the figures
#   4. the teardown
#
# THE TEARDOWN COMES LAST. Round 1 destroyed its box two hours before the
# scores arrived, and the card's own rule — add arms when the scores show a
# direction — needed a live box at that moment. So the boxes stay up until
# every score file exists.
#
# THE BUDGET. vast.ai holds $18.59 and this round may spend $12. Three boxes at
# $0.3356/h is $1.0068/h, so the ceiling below is in hours of box life. A
# watchdog tears the boxes down when it is reached, whatever stage is running.
#
# Usage:
#   nohup setsid bash scripts/round2.sh > results/round2.log 2>&1 &
#   ARMS="a085 a095 s08b" bash scripts/round2.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"

# The three arms of this round, and the box each one takes.
JOBS="${JOBS:-box_b:a085 box_c:a095 box_d:s08b}"
STOP="${STOP:-$CF404_STOPS}"
MAIN_DIR="${MAIN_DIR:-$HOME/cf404_sync/box_a}"
MAIN_ROOT="$MAIN_DIR/sync"
POLL="${POLL:-300}"
# 10.5 hours of three boxes at $0.3356/h is $10.57, under the $12 the round is
# allowed. The round-1 box needed 6 h 9 m for four arms including its bootstrap.
DEADLINE_HOURS="${DEADLINE_HOURS:-10.5}"
mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"

LOG="$CF404_RESULTS/round2.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 round2] $*" | tee -a "$LOG"; }

read -r -a job_list <<<"$JOBS"
arms=""; labels=""
for job in "${job_list[@]}"; do
  labels="$labels ${job%%:*}"; arms="$arms ${job##*:}"
  cf404_require_arm "${job##*:}" || exit $?
done
arms="${arms# }"; labels="${labels# }"

# ---- the teardown, which every exit path runs -------------------------------
#
# Only an instance THIS round provisioned is ever destroyed, and only by the id
# its own `.env` file records. `vastrun-destroy` takes the id and the label
# together as a confirmation token, so an id that no longer carries this
# round's label is refused. The vast.ai account is shared with other sessions
# (CLAUDE.md).
teardown(){
  local label envf inst
  for label in $labels; do
    envf="$CF404_RESULTS/round2_${label}.env"
    [ -s "$envf" ] || { say "teardown: no address for $label"; continue; }
    inst="$(awk -F= '$1=="INSTANCE"{print $2}' "$envf")"
    [ -n "$inst" ] || { say "teardown: no instance id in $envf"; continue; }
    say "teardown: destroying $inst (cf404-${label//_/-})"
    timeout 300 vastrun-destroy "$inst" "cf404-${label//_/-}" 2>&1 \
      | sed 's/^/  /' | tee -a "$LOG"
    say "teardown: stopping the sync loop of $label" \
        "($(cf404_stop_sync_loop "$HOME/cf404_sync/$label") loop(s))"
  done
  timeout 120 vastrun-status 2>&1 | sed 's/^/  /' | tee -a "$LOG"
}

# ---- the watchdog -----------------------------------------------------------
#
# It holds no other state, so it survives every failure of the stages below. A
# stage that hangs on a dead box would otherwise bill until a person looks.
watchdog(){
  local secs
  secs="$(awk -v h="$DEADLINE_HOURS" 'BEGIN{printf "%d", h*3600}')"
  sleep "$secs"
  say "WATCHDOG: ${DEADLINE_HOURS} h reached — tearing the boxes down"
  teardown
}
watchdog & WATCHDOG=$!
stop_watchdog(){ kill -TERM "$WATCHDOG" 2>/dev/null; }

say "START arms='$arms' boxes='$labels' deadline=${DEADLINE_HOURS}h"
say "credit before: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"

# ---- 1: the three boxes -----------------------------------------------------
pids=(); names=()
for job in "${job_list[@]}"; do
  label="${job%%:*}"; arm="${job##*:}"
  if [ -f "$CF404_RESULTS/round2_${label}.done" ]; then
    say "$label/$arm already finished"
    continue
  fi
  say "starting $label for $arm"
  nohup bash "$HERE/round2_box.sh" "$label" "$arm" \
    >>"$CF404_RESULTS/round2_${label}.out" 2>&1 &
  pids+=($!); names+=("$label/$arm")
  # Three cold HuggingFace readers on one account open together otherwise.
  sleep "${STAGGER:-120}"
done

failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  say "box ${names[$i]} rc=$rc"
  [ $rc -eq 0 ] || failed=$(( failed + 1 ))
done
say "$failed of ${#pids[@]} box driver(s) failed"

# ---- 2: the GIFT-Evals ------------------------------------------------------
#
# One eval per arm whose head landed. `evals_elisa.sh` skips an arm with no
# head and an arm already scored, so a failed box costs only its own arm.
say "starting the GIFT-Evals for '$arms'"
ARMS="$arms" CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" \
  bash "$HERE/evals_elisa.sh" >>"$CF404_RESULTS/evals_round2.out" 2>&1
say "evals rc=$?"
for arm in $arms; do
  f="$(cf404_score_file "$arm" "$STOP")"
  if [ -s "$f" ]; then say "score $arm $(tr -d ' \t\r\n' <"$f")"
  else say "score $arm MISSING"; fi
done

# ---- 3: the artefacts, the shard check and the figures ----------------------
say "shard check"
python3 "$HERE/check_shards.py" --root "$MAIN_ROOT" \
  --out "$CF404_RESULTS/shard_check.txt" 2>&1 | tail -20 | tee -a "$LOG"
say "report_assets"
CF404_ROOT="$MAIN_ROOT" CF404_SYNC_ROOT="$MAIN_ROOT" CF404_SYNC_DIR="$MAIN_DIR" \
  bash "$HERE/report_assets.sh" >>"$CF404_RESULTS/report_assets_round2.out" 2>&1
say "make_plots"
CF404_ROOT="$MAIN_ROOT" bash "$HERE/make_plots.sh" \
  >>"$CF404_RESULTS/make_plots_round2.out" 2>&1
say "plots rc=$?"

# ---- 4: the teardown --------------------------------------------------------
#
# Every score that exists, exists now. Gap 6 of the review is closed here: the
# boxes outlived the scores.
scored=0
for arm in $arms; do
  [ -s "$(cf404_score_file "$arm" "$STOP")" ] && scored=$(( scored + 1 ))
done
say "$scored of $(echo "$arms" | wc -w) arm(s) scored — tearing the boxes down"
teardown
stop_watchdog
say "credit after: $(timeout 90 vastrun-balance 2>&1 | tr '\n' ' ')"
say "ROUND 2 DONE"
