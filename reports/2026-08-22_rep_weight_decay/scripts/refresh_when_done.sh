#!/bin/bash
# #409 — wait for the arms still in flight, then rebuild every artefact.
#
# WHY THIS SCRIPT EXISTS. Gap 4 of the run review was stale artefacts: the
# launcher died in the Hub outage of 08-23 and `results/RUN_STATE.md` froze with
# three scores against six. The same thing happens again whenever an arm lands
# and nobody re-runs the figures.
#
# `dec_ramp30k_m080` started on 08-25 at 16:14 and takes about seven hours of
# backbone, then a head and a 97-config eval. So this waits for it and rebuilds.
#
# IT WRITES ARTEFACTS, NEVER TRAINING. It runs `make_plots.sh` and
# `run_state.sh`, which read the checkpoint root and regenerate the figures and
# the tables. Both are idempotent, so a second agent running them at the same
# time changes nothing. It starts no leg, it kills nothing, and it does NOT
# commit: a background `git` beside a live session is how two sessions lose
# each other's work.
#
# Usage:
#   nohup setsid bash scripts/refresh_when_done.sh >/dev/null 2>&1 &
#   DEADLINE_H=12 POLL=600 bash scripts/refresh_when_done.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

POLL="${POLL:-600}"
DEADLINE_H="${DEADLINE_H:-14}"
LOG="$CF409_RESULTS/refresh_when_done.log"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409 refresh] $*" | tee -a "$LOG"; }

# An arm of this study still training. Matches the trainer command line on its
# checkpoint root, so another study's leg never counts.
legs_up(){
  pgrep -fa 'scripts/train.py' 2>/dev/null \
    | grep -c -- "--save-dir $CF409_ROOT" || true
}

# Arms with a losses CSV but no score file yet.
unscored(){
  local arm n=0
  for arm in $CF409_ARMS; do
    [ -s "$CF409_RESULTS/score_${arm}_bb40k_h30k_student.txt" ] && continue
    [ -n "$(cf409_losses_csvs "$arm" "${CF409_STOPS%% *}" | head -1)" ] \
      && n=$((n + 1))
  done
  echo "$n"
}

rebuild(){  # <note>
  say "rebuilding — $1"
  bash "$HERE/make_plots.sh" >>"$LOG" 2>&1 \
    && say "make_plots.sh ok" || say "WARN make_plots.sh rc=$?"
  bash "$HERE/run_state.sh" "$1" >>"$LOG" 2>&1 \
    && say "run_state.sh ok" || say "WARN run_state.sh rc=$?"
  say "artefacts refreshed. Commit them from a session, not from here."
}

deadline=$(( $(date +%s) + DEADLINE_H * 3600 ))
say "START poll=${POLL}s deadline=${DEADLINE_H}h legs_up=$(legs_up)"

while [ "$(date +%s)" -lt "$deadline" ]; do
  up="$(legs_up)"
  if [ "$up" -eq 0 ]; then
    say "no leg of this study runs. $(unscored) arm(s) still unscored."
    rebuild "every leg stopped; $(unscored) arm(s) unscored"
    exit 0
  fi
  say "$up leg(s) up, $(unscored) unscored — waiting ${POLL}s"
  sleep "$POLL"
done

say "DEADLINE after ${DEADLINE_H}h with $(legs_up) leg(s) still up"
rebuild "deadline reached; a leg still runs"
exit 0
