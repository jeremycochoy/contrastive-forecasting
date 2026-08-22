#!/bin/bash
# #407 — the read-back, in one place, and out of an agent's hands.
#
# A band drains hours after the agent that fired it has gone. Nothing then
# brings its numbers into the checkout, regenerates the figure or refreshes
# the mirror, so the study keeps the PREVIOUS band's numbers and the figure
# goes stale.
#
# Round 3 put this behind `await_redraw.sh`, a harness background task. That
# task died with its session and its read-back never ran. So the read-back
# lives here now, and two things that outlive an agent call it:
#
#   watchdog.sh         every hourly tick, for whatever drained since.
#   replicate_heads.sh  the moment its own band drains.
#
# Every step is idempotent and cheap. A stop with no new draw re-writes the
# same numbers. A pair whose eval holds fewer than 97 configs does not cross,
# so a half-finished band cannot enter a figure.
#
# Usage: [WT=<checkout>] [CF373_ROOT=<durable root>] read_back.sh [stop_k ...]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES" "$STUDY/plots"

LOG="$RES/read_back.log"
log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-readback] $*" | tee -a "$LOG"; }

STOPS_K=("$@")
[ "${#STOPS_K[@]}" -gt 0 ] || STOPS_K=(200 300 450 665)

fail=0
step(){ # <name> <command...>
  local name="$1"; shift
  "$@" || { log "WARN: $name rc=$?"; fail=1; }
}

# `collect.sh` carries the driver's own six pairs, whose tags hold no seed.
# `collect_replicates.sh` carries the `_s<seed>` draws. Round 7 found the
# first one missing here: the 665k student pair reached the study only
# because an agent ran it by hand, and the 665k teacher pair would have
# needed the same. Both are idempotent and both gate on 97 configs.
step collect bash "$HERE/collect.sh" >>"$LOG" 2>&1
step collect_replicates bash "$HERE/collect_replicates.sh" "${STOPS_K[@]}" >>"$LOG" 2>&1
step head_band python3 "$HERE/head_band.py" --csv "$RES/head_band.csv" \
  >"$RES/head_band.txt" 2>>"$LOG"
step teacher_track python3 "$HERE/teacher_frozen_track.py" --csv "$RES/teacher_frozen_track.csv" \
  >"$RES/teacher_frozen_track.txt" 2>>"$LOG"
step plot python3 "$HERE/plot_full_pass.py" --out "$STUDY/plots/full_pass.png" \
  >>"$LOG" 2>&1
step mirror bash "$HERE/mirror_durable.sh" >>"$LOG" 2>&1

log "read-back done for stops ${STOPS_K[*]} (fail=$fail)"
exit "$fail"
