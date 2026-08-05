#!/bin/bash
# #393 — one line per thing worth knowing about, for a Monitor to consume.
#
# Usage:  bash scripts/watch_events.sh
#
# The ladder runs for hours across seven machines. Polling it in a loop burns
# a session on nothing; this emits an event only when something changes, and
# an hourly line so silence can be told apart from a dead watcher.
#
# Silence is not success, so the failure signatures are watched as loudly as
# the scores: a driver that exits, a Traceback or CUDA error in any run log,
# a box that stops answering. A watcher that only greps for DONE reads exactly
# the same as one watching a crashloop.
#
# Live sources only. `results/ladder_all.csv` is rebuilt every 30 min by the
# artefact loop, far too slow to notice a score; the broker log is written the
# moment an eval finishes and elisa's own ladder.csv the moment a stop closes.
set -uo pipefail
EXP="${EXP:-/tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder}"
RES="$EXP/results"
INTERVAL="${WATCH_INTERVAL:-300}"
HEARTBEAT="${WATCH_HEARTBEAT:-3600}"
STATE=$(mktemp -d); trap 'rm -rf "$STATE"' EXIT

ev(){ echo "[$(date -u '+%H:%M')Z] $*"; }

# `grep -c` and `pgrep -c` print 0 on no match and STILL exit 1. A trailing
# `|| echo 0` therefore appends a second line and every later [ ] comparison
# fails with "integer expression expected". Count through these two instead.
cnt(){  local n; n=$(grep -cE "$1" "$2" 2>/dev/null); echo "${n:-0}"; }
pcnt(){ local n; n=$(pgrep -fc "$1" 2>/dev/null); echo "${n:-0}"; }

# Lines already reported, so a restart of the source file cannot replay them.
seen_init(){  # <tag> <file> <regex>
  local tag="$1" f="$2"
  cnt "$3" "$f" > "$STATE/$tag"
}
seen_new(){  # <tag> <file> <regex>  -> prints the new matching lines
  local tag="$1" f="$2" re="$3" had now
  [ -f "$f" ] || return 0
  had=$(cat "$STATE/$tag" 2>/dev/null || echo 0)
  now=$(cnt "$re" "$f")
  [ "$now" -gt "$had" ] && grep -E "$re" "$f" | tail -n $(( now - had ))
  echo "$now" > "$STATE/$tag"
  return 0
}

BROKER="$RES/eval_broker.log"
RELEASE="$RES/release.log"
LADDER="$RES/ladder.csv"
EXTEND="$RES/extend.log"
RUNS="${CF393_RUNS:-/home/jupyter/checkpoints_backup/cf-393}"

# Score files on disk, counted directly. The two log sources above miss two
# cases and both are now normal: a stop finished by adopt_cell.sh has no
# driver to write elisa's ladder.csv, and a broker eval whose box was
# released cannot log DONE because the push back fails. The number is still
# measured and on disk, and scripts/scores_from_evals.py will pool it, so
# the watch has to see it.
score_files(){ ls "$RUNS"/*/eval/score_bb*.txt "$RUNS"/_broker/*/*/*/score.txt 2>/dev/null | wc -l; }

seen_init broker  "$BROKER"  'DONE score='
seen_init release "$RELEASE" 'RELEASED|destroying|vastrun-destroy failed'
seen_init extend  "$EXTEND"  'HOLD_ABOVE|launched|ABORT'
score_files > "$STATE/scorefiles"
[ -f "$LADDER" ] && wc -l < "$LADDER" > "$STATE/ladder" || echo 0 > "$STATE/ladder"
for f in "$RES"/run_cf393_*.log; do
  [ -f "$f" ] && cnt 'Traceback|CUDA error|AcceleratorError|out of memory' "$f" \
    > "$STATE/err_$(basename "$f" .log)"
done
drivers_before=$(pcnt 'ladder\.py --cells')
ev "watch armed: $drivers_before driver(s) on elisa, heartbeat every $((HEARTBEAT/60))m"
last_hb=$(date +%s)

while :; do
  sleep "$INTERVAL"

  # --- scores, from the two live sources ---------------------------------
  while read -r l; do [ -n "$l" ] && ev "SCORE ${l#*\] }"; done \
    < <(seen_new broker "$BROKER" 'DONE score=')
  now_l=$(wc -l < "$LADDER" 2>/dev/null || echo 0)
  had_l=$(cat "$STATE/ladder" 2>/dev/null || echo 0)
  if [ "$now_l" -gt "$had_l" ]; then
    tail -n $(( now_l - had_l )) "$LADDER" | while read -r r; do ev "SCORE elisa $r"; done
    echo "$now_l" > "$STATE/ladder"
  fi

  # --- score files, whether or not a log announced them --------------------
  now_s=$(score_files)
  had_s=$(cat "$STATE/scorefiles" 2>/dev/null || echo 0)
  if [ "$now_s" -gt "$had_s" ]; then
    ls -t "$RUNS"/*/eval/score_bb*.txt "$RUNS"/_broker/*/*/*/score.txt 2>/dev/null \
      | head -n $(( now_s - had_s )) \
      | while read -r p; do
          ev "SCORE $(sed -E 's|.*cf-393/||; s|/eval/score_|  |; s|\.txt$||; s|/score\.txt$||' <<<"$p") = $(cat "$p" 2>/dev/null)"
        done
    echo "$now_s" > "$STATE/scorefiles"
  fi

  # --- releases -----------------------------------------------------------
  while read -r l; do [ -n "$l" ] && ev "BOX ${l#*\] }"; done \
    < <(seen_new release "$RELEASE" 'RELEASED|destroying|vastrun-destroy failed')

  # --- the extension supervisor -------------------------------------------
  while read -r l; do [ -n "$l" ] && ev "EXTEND ${l#*\] }"; done \
    < <(seen_new extend "$EXTEND" 'HOLD_ABOVE|launched|ABORT')

  # --- failures in any run log, elisa's own and every synced copy ----------
  for f in "$RES"/run_cf393_*.log; do
    [ -f "$f" ] || continue
    tag="err_$(basename "$f" .log)"
    had=$(cat "$STATE/$tag" 2>/dev/null || echo 0)
    now=$(cnt 'Traceback|CUDA error|AcceleratorError|out of memory' "$f")
    if [ "$now" -gt "$had" ]; then
      ev "FAIL $(basename "$f" .log): $(grep -E 'Traceback|CUDA error|AcceleratorError|out of memory' "$f" | tail -1 | cut -c1-110)"
    fi
    echo "$now" > "$STATE/$tag"
  done

  # --- a driver leaving -----------------------------------------------------
  # Expected at the HOLD_ABOVE ceiling, so this reports rather than alarms;
  # which of the two it is shows in the cell's own log.
  drivers_now=$(pcnt 'ladder\.py --cells')
  if [ "$drivers_now" -ne "$drivers_before" ]; then
    ev "DRIVERS on elisa: $drivers_before -> $drivers_now"
    drivers_before=$drivers_now
  fi

  # --- heartbeat ------------------------------------------------------------
  nowt=$(date +%s)
  if [ $(( nowt - last_hb )) -ge "$HEARTBEAT" ]; then
    last_hb=$nowt
    bal=$(vastrun-balance 2>/dev/null | awk '/Credit/{print $2}')
    box=$(vastrun-status 2>/dev/null | grep -c running)
    ev "HEARTBEAT scored=$(( $(wc -l < "$RES/ladder_all.csv" 2>/dev/null || echo 1) - 1 ))" \
       "credit=${bal:-?} boxes=${box:-?} drivers=$drivers_now" \
       "evals=$(pcnt 'eval_gift_eval_official')"
  fi
done
