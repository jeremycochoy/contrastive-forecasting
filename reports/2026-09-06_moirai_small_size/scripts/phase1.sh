#!/bin/bash
# #412 — the arms, one stop at a time, then a head and a score for each.
#
# The order is by STOP and not by arm. Every arm reaches its first stop before
# any arm starts on the next one, so a session that runs out of time holds the
# same stop on every arm rather than one finished arm and five empty ones.
# That is #393's spend order, and it is what makes a partial card readable.
#
# A leg resumes the arm's FURTHEST checkpoint with its optimizer state
# (`newest_ckpt`, #393), so the stops are one continuous run for each arm. A
# stop this script never asks for costs nothing later: `STOPS=200000` on an
# arm that holds a 40,000-step checkpoint trains the 160,000 steps between
# them and skips the 100,000-step head and its 97 GIFT-Eval configs.
#
# ---- The two knobs, and the run plan they carry -------------------------------
#
# `ARMS` and `STOPS` select the legs. The plan of PR #413 runs in two passes:
#
#   1. the five mandatory arms and the repeat seed, to 40,000 steps ONLY.
#      Two cards, two arms on each at most, because both GPUs of this box
#      carry other work.
#
#        Card A: k32_r100_09, then k32_r100_09_dec, its decay twin.
#        Card B: k3_r100_09, k3_r100_09b, k3_r100_09_dec, k32_r200_08.
#        k8_r100_09 is optional and goes last, on an idle card.
#
#      This order completes the decay pair and the seed band first, so an
#      interruption still leaves an answer. Cost about 27 GPU-hours.
#
#   2. the gate, on the 40,000-step GIFT scores. Then `STOPS=200000` on the
#      arms that pass it. `run.sh` holds the gate rule.
#
# Each stage is idempotent. An arm whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes for
# each shard. So a re-run after a crash costs only what did not finish.
#
# ---- An arm that lost the contrastive task does not climb ---------------------
#
# `run_arm.sh` exits CF412_RC_COLLAPSED (4) when the AUC gate stopped the leg.
# A higher stop of that arm would train the same collapse, so this script drops
# the arm from every stop above the one it lost, and gives it no head. The arm
# still has its whole AUC curve and its loss by term, which is the answer the
# card asks for.
#
# Usage:  BB_GPU=0 bash scripts/phase1.sh
#         BB_GPU=0 STOPS=40000 ARMS="k32_r100_09 k32_r100_09_dec" \
#           bash scripts/phase1.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-0}"
ARMS="${ARMS:-$CF412_ARMS}"
STOPS="${STOPS:-$CF412_STOPS}"
mkdir -p "$CF412_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#412 phase1] $*" \
  | tee -a "$CF412_RESULTS/phase1.log"; }

log "arms: $ARMS"
log "stops: $STOPS  gpu: $BB_GPU"

failed=0
collapsed=""
for stop in $STOPS; do
  for arm in $ARMS; do
    cf412_require_arm "$arm" || exit $?
    cf412_require_stop "$stop" || exit $?
    cf412_is_in "$arm" "$collapsed" && {
      log "backbone $arm SKIPPED at $stop — it lost the contrastive task"
      continue; }
    log "backbone $arm -> $stop"
    BB_GPU="$BB_GPU" bash "$HERE/run_arm.sh" "$arm" "$stop"
    rc=$?
    if [ "$rc" -eq "$CF412_RC_COLLAPSED" ]; then
      log "backbone $arm LOST the contrastive task — see" \
          "$(cf412_collapse_file "$arm")"
      collapsed="$collapsed $arm"
      continue
    fi
    [ "$rc" -eq 0 ] || {
      log "backbone $arm stop $stop FAILED rc=$rc"
      failed=$(( failed + 1 )); continue; }
    log "head $arm bb$stop"
    BB_GPU="$BB_GPU" bash "$HERE/head_eval.sh" "$arm" "$stop" || {
      log "head $arm stop $stop FAILED"; failed=$(( failed + 1 )); }
  done
done

[ -n "$collapsed" ] && log "lost the contrastive task:$collapsed"
log "phase1 done — $failed failure(s)"
[ "$failed" -eq 0 ] || exit 1
