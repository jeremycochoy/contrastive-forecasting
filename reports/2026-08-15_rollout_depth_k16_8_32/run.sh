#!/bin/bash
# #401 — the whole study on ONE machine, in the order the card runs it.
#
#   1. smoke     k = 0, 8 and 32, a few hundred steps each, under this
#                study's own objective. Records the step time and the peak
#                GPU memory of THIS run's process tree. The run plan depends
#                on both, so it runs first and the numbers land in
#                results/<reduce>/smoke_depth.csv.
#   2. trial     the whole pipeline at a few hundred steps and one GIFT-Eval
#                config, on a trial root. The head half of this study costs
#                14 hours of backbone time before it runs for the first time,
#                so it runs here first, in minutes.
#   3. phase 1   two arms, k = 8 and k = 32, each to 40k / 100k / 200k
#                backbone steps. One student head at 30,000 steps per stop,
#                then that head's 97-config GIFT-Eval.
#   4. phase 2   the two best arms again, with the head budget matched to the
#                backbone stop: 40k / 100k / 200k head steps.
#   5. plots     the card's two deliverables, from whatever is scored.
#
# THE STUDY DOES NOT RUN THIS WAY. It runs on four GPUs across two machines,
# because that is the faster answer:
#
#   rented box, 2 GPUs   scripts/provision_box.sh, scripts/bootstrap_box.sh,
#                        then scripts/launch_box.sh — one backbone arm per
#                        card, no heads.
#   elisa, GPU 0         sync/launch_sync.sh to pull the box's checkpoints,
#                        then scripts/launch_elisa.sh — every head, every
#                        97-config GIFT-Eval, every figure.
#
# This file stays because the trial runs through it, and because one machine
# with two free cards can run the whole card without the box.
#
# Each stage is idempotent. A stop whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes per
# shard. So a re-run after a crash costs only what did not finish.
#
# Usage:  BB_GPU=0 bash run.sh              # everything
#         BB_GPU=0 bash run.sh phase1       # one stage
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${*:-smoke trial phase1 phase2 plots}"

for stage in $STAGES; do
  case "$stage" in
    smoke)  bash "$HERE/scripts/smoke_depth.sh" "${SMOKE_STEPS:-300}" ;;
    trial)  bash "$HERE/scripts/trial_head.sh" ;;
    phase1) bash "$HERE/scripts/phase1.sh" ;;
    phase2) bash "$HERE/scripts/phase2.sh" ;;
    plots)  bash "$HERE/scripts/make_plots.sh" ;;
    *) echo "ABORT: unknown stage '$stage'" >&2
       echo "  (smoke trial phase1 phase2 plots)" >&2; exit 2 ;;
  esac
  rc=$?
  [ $rc -eq 0 ] || { echo "ABORT: stage '$stage' rc=$rc" >&2; exit $rc; }
done
