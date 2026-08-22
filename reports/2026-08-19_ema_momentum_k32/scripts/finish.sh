#!/bin/bash
# #404 — the last stage, detached from any session.
#
# `evals_elisa.sh` runs the four 97-config GIFT-Evals and writes one
# `score_<tag>.txt` per arm. This waits for it, then makes the artefacts the
# card asks for. It holds no SSH and no GPU, so it survives a dead session.
#
#   1. wait for the eval driver to exit
#   2. `report_assets.sh` — the eval CSV, the summary, the log and the
#      downsampled loss curve, into the study directory
#   3. `make_plots.sh` — collect, then the card's four deliverables
#
# CF404_ROOT. The default root is the checkpoint store, which this study does
# not use: the four backbones and the four heads live in the sync tree. Both
# stages take that tree, or `collect.sh` finds no eval CSV and the radar has
# no rows.
#
# Usage:
#   nohup setsid bash scripts/finish.sh > results/finish.log 2>&1 &
#   WAIT_PID=<pid> bash scripts/finish.sh   # wait on one eval driver
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"

ROOT="${CF404_ROOT_GIVEN:-$CF404_SYNC_ROOT}"
POLL="${POLL:-120}"
TIMEOUT="${TIMEOUT:-28800}"      # 8 h ceiling on the eval wait
mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 finish] $*"; }

# ---- 1: wait for the eval driver --------------------------------------------
# By PID when the caller knows it, else by name. An eval driver that is
# already gone is not an error: the scores are then on disk and stage 2 runs.
waited=0
while :; do
  if [ -n "${WAIT_PID:-}" ]; then
    kill -0 "$WAIT_PID" 2>/dev/null || { say "eval driver $WAIT_PID gone"; break; }
  else
    pgrep -f 'bash .*[e]vals_elisa\.sh' >/dev/null || { say "no eval driver"; break; }
  fi
  if [ "$waited" -ge "$TIMEOUT" ]; then
    say "TIMEOUT after ${waited}s — going on with what is scored"
    break
  fi
  [ $(( waited % 1800 )) -eq 0 ] && say "waiting, ${waited}s so far — $(
    ls "$CF404_RESULTS"/score_*.txt 2>/dev/null | wc -l) score(s) written"
  sleep "$POLL"; waited=$(( waited + POLL ))
done

for arm in $CF404_ARMS; do
  f="$(cf404_score_file "$arm" "$CF404_STOPS")"
  if [ -s "$f" ]; then say "score $arm $(tr -d ' \t\r\n' <"$f")"
  else say "score $arm MISSING"; fi
done

# ---- 2 and 3: the artefacts and the four deliverables ------------------------
say "report_assets"
CF404_ROOT="$ROOT" bash "$HERE/report_assets.sh"
say "make_plots"
CF404_ROOT="$ROOT" bash "$HERE/make_plots.sh"
say "FINISH DONE"
