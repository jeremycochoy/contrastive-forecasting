#!/bin/bash
# #412 — the arms, one stop at a time, then a head and a score for each.
#
# The order is by STOP and not by arm. Every arm reaches 40,000 steps before
# any arm starts on 100,000, so a session that runs out of time holds the same
# stop on every arm rather than one finished arm and four empty ones. That is
# #393's spend order, and it is what makes a partial card readable.
#
# A leg resumes the arm's furthest checkpoint with its optimizer state, so the
# stops are one continuous run for each arm, not three runs.
#
# Each stage is idempotent. An arm whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes for
# each shard. So a re-run after a crash costs only what did not finish.
#
# Usage:  BB_GPU=0 bash scripts/phase1.sh
#         STOPS=40000 ARMS="k3_r100_09 k32_r100_09" bash scripts/phase1.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-0}"
ARMS="${ARMS:-$CF412_ARMS}"
STOPS="${STOPS:-$CF412_STOPS}"
mkdir -p "$CF412_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 phase1] $*" \
  | tee -a "$CF412_RESULTS/phase1.log"; }

failed=0
for stop in $STOPS; do
  for arm in $ARMS; do
    cf412_require_arm "$arm" || exit $?
    cf412_require_stop "$stop" || exit $?
    log "backbone $arm -> $stop"
    BB_GPU="$BB_GPU" bash "$HERE/run_arm.sh" "$arm" "$stop" || {
      log "backbone $arm stop $stop FAILED"; failed=$(( failed + 1 )); continue; }
    log "head $arm bb$stop"
    BB_GPU="$BB_GPU" bash "$HERE/head_eval.sh" "$arm" "$stop" || {
      log "head $arm stop $stop FAILED"; failed=$(( failed + 1 )); }
  done
done

log "phase1 done — $failed failure(s)"
[ "$failed" -eq 0 ] || exit 1
