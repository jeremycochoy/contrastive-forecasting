#!/bin/bash
# #412 — a status line every 15 minutes, and a durable copy of results/.
#
# `results/` sits in a worktree under /tmp. The checkpoints, the losses CSVs
# and the eval output are durable, but the score files and the logs are not.
# This mirrors them, so a lost worktree costs no measurement.
#
# It also re-runs `collect.sh`, `make_plots.sh` and `report_tables.py` on every
# tick. Every table and every figure of the card is built from the durable
# artefacts, so none of them is ever older than one tick. The first pass wrote
# the tables four minutes after the launch, when no arm held a score and no
# losses CSV held a row above the AUC warm-up.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"
MIRROR="${CF412_MIRROR:-/home/jupyter/cf412_mirror}"
EVERY="${CF412_HEARTBEAT_EVERY:-900}"
mkdir -p "$MIRROR"
while :; do
  bash "$HERE/watch_status.sh" >>"$CF412_RESULTS/watch.log" 2>&1
  bash "$HERE/collect.sh" >>"$CF412_RESULTS/collect.log" 2>&1
  bash "$HERE/make_plots.sh" >>"$CF412_RESULTS/collect.log" 2>&1
  python3 "$HERE/report_tables.py" >>"$CF412_RESULTS/collect.log" 2>&1
  cp -a "$CF412_RESULTS/." "$MIRROR/" 2>/dev/null
  sleep "$EVERY"
done
