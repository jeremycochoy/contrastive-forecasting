#!/bin/bash
# #373 review gap 3 — block until the two `G_B1_k0_aw4` scores land, or until
# the work stalls.
#
# `gap_watch.sh` prints an event stream for a human. This one is the machine
# gate: it exits 0 only when both scores exist, and it exits non-zero when
# progress stops. The agent waits on this exit code instead of polling.
#
# A tail of a log is not a liveness probe. A hung trainer keeps its log file
# and its process, and both look healthy. Each probe here demands that a
# COUNTER MOVED since the previous probe:
#
#   phase A (backbone)      the step counter in the run log
#   phase B (heads, evals)  the finished GIFT-Eval configs, or a head process
#
# STALL_S of no movement in either counter ends the wait with rc=3, so a
# silent stall surfaces instead of running out the clock.
#
# Usage: nohup bash scripts/gap3_await.sh > results/gap3_await.out 2>&1 &
#   rc=0  both scores on disk
#   rc=2  a run died
#   rc=3  no counter moved for STALL_S
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES="$HERE/../results"
WRES="${WRES:-/home/jupyter/wt-cf-373-train/reports/2026-08-08_rollout_depth/results}"
CF373_ROOT="${CF373_ROOT:-/home/jupyter/checkpoints_backup/cf-373}"
NAME="bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090_cf373k0_aw4"
BBLOG="$WRES/run_${NAME}.log"
CK="$CF373_ROOT/$NAME/${NAME}_40k.pth"
S3S="$WRES/score_G_B1_k0_aw4_bb40k_student.txt"
S3T="$WRES/score_G_B1_k0_aw4_bb40k_teacher.txt"
DONE="$RES/gap3_await.done"

PROBE_S="${PROBE_S:-300}"
STALL_S="${STALL_S:-5400}"     # 90 min. One eval shard takes well under that.

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [await] $*" | tee -a "$RES/gap3_await.log"; }

evald(){ # <tag> -> finished configs
  local g="$CF373_ROOT/eval/$1/gift" n=0 d
  for d in "$g"/shard_*; do
    [ -d "$d" ] || continue
    n=$(( n + $(grep -ac 'MASE=' "$d/shard.log" 2>/dev/null || echo 0) ))
  done
  echo "$n"
}

bbstep(){ grep -oE '^\[ *[0-9]+\]' "$BBLOG" 2>/dev/null | tail -1 | tr -dc '0-9'; }

finish(){ # <rc> <message>
  log "$2"
  { echo "rc=$1"; echo "$2";
    echo "student=$(cat "$S3S" 2>/dev/null || echo MISSING)"
    echo "teacher=$(cat "$S3T" 2>/dev/null || echo MISSING)"; } >"$DONE"
  exit "$1"
}

log "waiting on $S3S and $S3T (probe ${PROBE_S}s, stall ${STALL_S}s)"
rm -f "$DONE"
last_move=$(date +%s); last_counter=""

while :; do
  if [ -s "$S3S" ] && [ -s "$S3T" ]; then
    finish 0 "BOTH SCORES IN: student $(cat "$S3S") teacher $(cat "$S3T")"
  fi

  if [ ! -f "$CK" ]; then
    # ---- phase A: the backbone ------------------------------------------
    counter="bb:$(bbstep)"
    if ! pgrep -f "run-name $NAME" >/dev/null 2>&1; then
      finish 2 "backbone process gone with no 40k checkpoint (last $counter)"
    fi
  else
    # ---- phase B: two heads, two evals ----------------------------------
    s=$(evald G_B1_k0_aw4_bb40k_student); t=$(evald G_B1_k0_aw4_bb40k_teacher)
    heads=$(pgrep -fc 'head_eval_bb.sh' 2>/dev/null || echo 0)
    counter="ev:$s/$t heads:$heads"
    # The driver owns the retries. It exiting with a score missing is terminal.
    if ! pgrep -f 'gap3_heads.sh' >/dev/null 2>&1; then
      finish 2 "gap3_heads.sh exited with a score missing (last $counter)"
    fi
  fi

  now=$(date +%s)
  if [ "$counter" != "$last_counter" ]; then
    last_counter="$counter"; last_move=$now
    log "$counter"
  elif [ $(( now - last_move )) -ge "$STALL_S" ]; then
    finish 3 "STALL: $counter unchanged for $(( (now - last_move) / 60 )) min"
  fi

  sleep "$PROBE_S"
done
