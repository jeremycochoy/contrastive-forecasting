#!/bin/bash
# #373 — one event stream for the two review-gap runs still in flight.
#
# Emits one line per thing worth acting on and exits when all three scores
# are in hand, or when a run dies. Silence means "still running": every
# terminal state below has a line, including the ones that are failures.
#
#   item 6   A3_k3_bb200k_student_s20260723   eval on the cores
#   item 3   G_B1_k0_aw4                      backbone, then 2 heads, then 2 evals
#
# Usage: bash scripts/gap_watch.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../results"
WRES="${WRES:-/home/jupyter/wt-cf-373-train/reports/2026-08-08_rollout_depth/results}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
NAME="bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0_aw4"
BBLOG="$WRES/run_${NAME}.log"
CK="$CF373_ROOT/$NAME/${NAME}_40k.pth"

S6="$WRES/score_A3_k3_bb200k_student_s20260723.txt"
S3S="$WRES/score_G_B1_k0_aw4_bb40k_student.txt"
S3T="$WRES/score_G_B1_k0_aw4_bb40k_teacher.txt"

# Local clock, to match every other log this experiment writes.
say(){ echo "[$(date '+%m-%d %H:%M')] $*"; }

evald(){ # <tag> -> configs finished
  # `grep -c` prints 0 AND returns 1 on no match. A `|| echo 0` fallback then
  # emits a second 0 and the arithmetic sees "0\n0". Use grep's own count.
  local g="$CF373_ROOT/eval/$1/gift" n=0 d c
  for d in "$g"/shard_*; do
    [ -d "$d" ] || continue
    c=$(grep -ac 'MASE=' "$d/shard.log" 2>/dev/null)
    n=$(( n + ${c:-0} ))
  done
  echo "$n"
}

last_bb=0; last_ev6=-1; last_tick=0; seen_ck=0
declare -A said

while :; do
  now=$(date +%s)

  # --- item 6: the reseed eval on the cores -------------------------------
  if [ -s "$S6" ]; then
    [ -n "${said[s6]:-}" ] || { say "ITEM6 SCORE A3 bb200k student reseed s20260723 = $(cat "$S6")"; said[s6]=1; }
  else
    n6=$(evald A3_k3_bb200k_student_s20260723)
    if [ "$n6" -ge 97 ] && [ "$last_ev6" -lt 97 ]; then say "ITEM6 eval 97/97 configs, aggregating"; fi
    if [ $(( n6 - last_ev6 )) -ge 25 ]; then say "ITEM6 eval $n6/97"; last_ev6=$n6; fi
  fi

  # --- item 3: backbone ---------------------------------------------------
  if [ "$seen_ck" -eq 0 ] && [ -f "$CK" ]; then
    say "ITEM3 backbone 40k checkpoint on disk ($(stat -c %s "$CK") bytes)"; seen_ck=1
  fi
  if [ "$seen_ck" -eq 0 ]; then
    step=$(grep -oE '^\[ *[0-9]+\]' "$BBLOG" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ -n "$step" ] || step=0
    if [ $(( step - last_bb )) -ge 10000 ]; then
      say "ITEM3 backbone $step/40000  $(grep -oE '[0-9.]+ sps  ETA [0-9.]+h' "$BBLOG" | tail -1)"
      last_bb=$step
    fi
    if ! pgrep -f "run-name $NAME" >/dev/null 2>&1 && [ ! -f "$CK" ]; then
      say "ITEM3 FAILED: backbone process gone with no 40k checkpoint"; exit 1
    fi
  fi

  # --- item 3: heads and evals -------------------------------------------
  for enc in student teacher; do
    sf="$WRES/score_G_B1_k0_aw4_bb40k_${enc}.txt"
    if [ -s "$sf" ]; then
      [ -n "${said[$enc]:-}" ] || { say "ITEM3 SCORE B1 k=0 L_align x4 bb40k $enc = $(cat "$sf")"; said[$enc]=1; }
    fi
  done

  # New driver lines: head start/end, aborts, give-ups.
  if [ -f "$RES/gaps_driver.log" ]; then
    grep -a '\[gap3\]' "$RES/gaps_driver.log" 2>/dev/null \
      | grep -aE 'HEAD (START|END)|GIVING UP|ABORT|TIMEOUT|backbone in hand|gap 3 done' \
      | while IFS= read -r line; do
          key=$(echo "$line" | md5sum | cut -c1-12)
          [ -f "/tmp/cf373_watch_$key" ] && continue
          : >"/tmp/cf373_watch_$key"
          say "$line"
        done
  fi

  # A crash inside a head or an eval.
  for f in "$CF373_ROOT/eval/G_B1_k0_aw4_bb40k_student/stop.log" \
           "$CF373_ROOT/eval/G_B1_k0_aw4_bb40k_teacher/stop.log" \
           "$CF373_ROOT/eval/A3_k3_bb200k_student_s20260723/stop.log"; do
    [ -f "$f" ] || continue
    grep -aE 'rc=[1-9]|ABORT|Traceback|CUDA out of memory' "$f" 2>/dev/null | tail -1 \
      | while IFS= read -r line; do
          key=$(echo "$f$line" | md5sum | cut -c1-12)
          [ -f "/tmp/cf373_watch_$key" ] && continue
          : >"/tmp/cf373_watch_$key"
          say "FAILURE $(basename "$(dirname "$f")"): $line"
        done
  done

  # --- done? --------------------------------------------------------------
  if [ -s "$S6" ] && [ -s "$S3S" ] && [ -s "$S3T" ]; then
    say "ALL THREE SCORES IN — item6 $(cat "$S6")  item3 student $(cat "$S3S")  teacher $(cat "$S3T")"
    exit 0
  fi

  # Half-hourly heartbeat so a stall is visible.
  if [ $(( now - last_tick )) -ge 1800 ]; then
    say "tick: bb $(grep -oE '^\[ *[0-9]+\]' "$BBLOG" 2>/dev/null | tail -1 | tr -dc '0-9')/40000  ev6 $(evald A3_k3_bb200k_student_s20260723)/97  ev3s $(evald G_B1_k0_aw4_bb40k_student)/97  ev3t $(evald G_B1_k0_aw4_bb40k_teacher)/97  load $(cut -d' ' -f1 /proc/loadavg)"
    last_tick=$now
  fi

  sleep 120
done
