#!/bin/bash
# #412 — the whole card on ONE machine, in the order it runs.
#
#   1. size     the parameter count of each candidate shape, and of the model
#               today. No GPU, seconds. It is the measurement the width rests
#               on.
#   2. smoke    each arm for a few tens of steps, through the same wrapper and
#               the same runner. It reads the shape back off the trainer's
#               command line, counts the depth columns, and records the step
#               time and the memory the run plan is sized from.
#   3. trial    one arm end to end: the backbone leg, its head, and ONE
#               GIFT-Eval config. The head trainer and the evaluation rebuild
#               the backbone from flags, so this is where a wrong width shows.
#   4. phase1   the arms of `ARMS` to each stop of `STOPS`, one 30,000-step
#               student head for each stop, then that head's 97 GIFT-Eval
#               configs. It also starts the AUC gate on every leg.
#   5. collect  the scores, the AUC verdicts and the loss by term, in three
#               tables.
#
# Each stage is idempotent. An arm whose checkpoint is on disk is a no-op, a
# head whose score file is written is a no-op, and a GIFT-Eval resumes for
# each shard. So a re-run after a crash costs only what did not finish.
#
# Both GPUs of this box carry other work, so an arm waits for a free card
# through #373's `gpu_gate.sh`, and no card takes more than two arms.
#
# ---- The measured cost -------------------------------------------------------
#
# `scripts/smoke.sh` measured this two times, 150 steps for each arm, while
# the card carried 4.0 GB and 80 to 97 percent of other work. The memory is a
# peak of a one-second sampler over this arm's OWN processes, so the other
# tenants of the card are not in it.
#
#   k     step time      memory              40,000 steps
#   3     272 ms         6,436 to 6,472 MiB  3.0 h
#   8     346 ms         5,424 to 7,160 MiB  3.8 h
#   32    651 to 688 ms  10,058 to 10,062 MiB  7.6 h
#
# THE STEP TIME IS AN UPPER BOUND. The card was busy through both runs. The
# parent cards measured 231 ms and 586 ms for k = 3 and k = 32 on a quieter
# card, which gives 2.6 h and 6.5 h. So the plan below costs 27 GPU-hours on
# a quiet card and up to 32 on this one.
#
# TWO k = 32 ARMS DO NOT SHARE ONE CARD HERE. 10.1 + 10.1 + 4.0 = 24.2 GB on
# a 24.0 GB card. Card A runs its two k = 32 arms one after the other, which
# is what the order below does.
#
# ---- The run plan, from PR #413 ----------------------------------------------
#
# STOP AT 40,000 STEPS FIRST. Do not launch 200,000 steps for every arm: that
# is 153 GPU-hours where the plan below is 38, or 64 if one k = 32 arm climbs.
#
#   Card A:  BB_GPU=0 STOPS=40000 ARMS="k32_r100_09 k32_r100_09_dec" \
#              bash run.sh phase1
#   Card B:  BB_GPU=1 STOPS=40000 \
#              ARMS="k3_r100_09 k3_r100_09b k3_r100_09_dec k32_r200_08" \
#              bash run.sh phase1
#
# `k8_r100_09` is optional. It goes last, and only on an idle card. This order
# completes the decay pair and the seed band first, so an interruption still
# leaves an answer. Cost about 27 GPU-hours.
#
# Seeds: 20260520 on every arm, 20260525 on `k3_r100_09b` alone. No new seed.
#
# ---- The gate, then 200,000 steps --------------------------------------------
#
# THE BAND is 0.0471 GM-Relative MASE, or the spread of the two `k3_r100_09`
# seeds, whichever is wider. #409 measured 0.0471 at this stop.
#
#   * Climb `k3_r100_09` ALWAYS. Its reference, 1.0651, is a 200,000-step
#     number, so the main question is answered at 200,000 steps and nowhere
#     else.
#   * Climb every other arm that sits within the band of the 40,000-step
#     leader.
#   * Allow `k32_r200_08` an extra 0.0291, the mid-ramp deficit its 1.1M twin
#     showed at 40,000 steps (1.1782 against 1.1491).
#   * THE PAIR. `k32_r100_09_dec` minus `k32_r100_09` measures the decay. If
#     that gap lands inside the band, repeat `k32_r100_09` once at a second
#     seed (6.5 h) instead of climbing both arms (52 h).
#
# A leg resumes the furthest checkpoint, so the gate adds no rerun cost. Each
# skipped stop also skips a 30,000-step head and 97 GIFT-Eval configs.
#
#   BB_GPU=0 STOPS=200000 ARMS="k3_r100_09" bash run.sh phase1 collect
#
# Usage:  BB_GPU=0 bash run.sh              # everything
#         BB_GPU=0 bash run.sh size smoke   # some stages
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES="${*:-size smoke trial phase1 collect}"

for stage in $STAGES; do
  case "$stage" in
    size)    bash "$HERE/scripts/size_table.sh" ;;
    smoke)   bash "$HERE/scripts/smoke.sh" ${SMOKE_STEPS:+"$SMOKE_STEPS"} ;;
    trial)   bash "$HERE/scripts/trial.sh" ;;
    phase1)  bash "$HERE/scripts/phase1.sh" ;;
    collect) bash "$HERE/scripts/collect.sh" ;;
    *) echo "ABORT: unknown stage '$stage'" >&2
       echo "  (size smoke trial phase1 collect)" >&2; exit 2 ;;
  esac
  rc=$?
  [ $rc -eq 0 ] || { echo "ABORT: stage '$stage' rc=$rc" >&2; exit $rc; }
done
