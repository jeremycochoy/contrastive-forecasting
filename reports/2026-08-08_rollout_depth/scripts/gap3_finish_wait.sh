#!/bin/bash
# #373 — block until item 3's two scores land, then exit.
#
# This is the SESSION's gate, not the run's. `gap3_await.sh` and
# `gap_watch.sh` are detached and outlive a session; this one is a child of
# the session, so its exit is the session's wake-up. It starts nothing and
# repairs nothing: `gap3_heads.sh` runs the heads and `gap3_supervise.sh`
# restarts that driver.
#
# It ends on one of three things and says which:
#   both scores on disk           rc=0
#   the eval counters stall       rc=2   (no config finished in STALL s)
#   the head driver and the
#   supervisor are both gone      rc=3
#
# Usage: nohup bash scripts/gap3_finish_wait.sh > results/gap3_finish_wait.log 2>&1 &
set -uo pipefail

WRES="${WRES:-/home/jupyter/wt-cf-373-train/reports/2026-08-08_rollout_depth/results}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
PROBE="${PROBE:-300}"
STALL="${STALL:-5400}"

S="$WRES/score_G_B1_k0_aw4_bb40k_student.txt"
T="$WRES/score_G_B1_k0_aw4_bb40k_teacher.txt"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [wait] $*"; }

evald(){ # <tag> -> configs finished
  # `grep -c` prints 0 AND returns 1 on no match, so take grep's own count
  # and default only when grep printed nothing at all.
  local g="$CF373_ROOT/eval/$1/gift" n=0 d c
  for d in "$g"/shard_*; do
    [ -d "$d" ] || continue
    c=$(grep -ac 'MASE=' "$d/shard.log" 2>/dev/null)
    n=$(( n + ${c:-0} ))
  done
  echo "$n"
}

say "waiting on $(basename "$S") and $(basename "$T") (probe ${PROBE}s, stall ${STALL}s)"
last_sum=-1
last_move=$(date +%s)

while :; do
  if [ -s "$S" ] && [ -s "$T" ]; then
    say "BOTH SCORES IN — student $(cat "$S")  teacher $(cat "$T")"
    exit 0
  fi

  s=$(evald G_B1_k0_aw4_bb40k_student)
  t=$(evald G_B1_k0_aw4_bb40k_teacher)
  sum=$(( s + t ))
  now=$(date +%s)

  if [ "$sum" -ne "$last_sum" ]; then
    say "ev student $s/97  teacher $t/97"
    last_sum=$sum
    last_move=$now
  elif [ $(( now - last_move )) -ge "$STALL" ]; then
    say "STALL: no config finished in $(( now - last_move ))s (student $s/97, teacher $t/97)"
    exit 2
  fi

  # A finished eval writes its score, so "no driver AND no score" is death,
  # not the normal end. The supervisor restarts the driver, so both must be
  # gone before this calls it.
  if ! pgrep -f 'gap3_heads.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'gap3_supervise.sh' >/dev/null 2>&1 \
     && ! pgrep -f 'head_eval_bb.sh' >/dev/null 2>&1; then
    say "DEAD: no head driver, no supervisor, no eval process, and a score is missing"
    exit 3
  fi

  sleep "$PROBE"
done
