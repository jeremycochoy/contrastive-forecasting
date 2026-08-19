#!/bin/bash
# #404 — the whole card on ONE machine, in the order it runs.
#
#   1. smoke    each arm for a few hundred steps, through the same wrapper and
#               the same runner. It reads the momentum back off the trainer's
#               command line, counts the depth columns, and records the step
#               time the run plan is sized from. Minutes, not hours.
#   2. phase1   the four arms to 40,000 backbone steps, one 30,000-step
#               student head per arm, then that head's 97 GIFT-Eval configs.
#   3. plots    the card's four deliverables, from whatever is scored.
#
# THE STUDY DOES NOT HAVE TO RUN THIS WAY. On two machines it runs:
#
#   rented box, 2 GPUs   scripts/launch_box.sh — two arms per card, backbones
#                        only.
#   elisa, GPU 0         sync/launch_sync.sh to pull the box's checkpoints,
#                        then scripts/launch_elisa.sh — every head, every
#                        97-config GIFT-Eval, every figure.
#
# This file stays because the smoke runs through it, and because one machine
# with a free card can run the whole card without the box.
#
# Each stage is idempotent. An arm whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes per
# shard. So a re-run after a crash costs only what did not finish.
#
# Usage:  BB_GPU=0 bash run.sh              # everything
#         BB_GPU=0 bash run.sh phase1       # one stage
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${*:-smoke phase1 plots}"

for stage in $STAGES; do
  case "$stage" in
    smoke)  bash "$HERE/scripts/smoke.sh" "${SMOKE_STEPS:-300}" ;;
    phase1) bash "$HERE/scripts/phase1.sh" ;;
    plots)  bash "$HERE/scripts/make_plots.sh" ;;
    *) echo "ABORT: unknown stage '$stage'" >&2
       echo "  (smoke phase1 plots)" >&2; exit 2 ;;
  esac
  rc=$?
  [ $rc -eq 0 ] || { echo "ABORT: stage '$stage' rc=$rc" >&2; exit $rc; }
done
