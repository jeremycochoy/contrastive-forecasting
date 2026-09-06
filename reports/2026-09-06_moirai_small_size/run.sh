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
#               configs. It also starts the AUC gate on every leg. `STOPS`
#               defaults to 40,000 steps, the first pass, and never to the
#               whole grid.
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
# `scripts/smoke.sh` ran each arm 150 steps and wrote `results/trial/smoke.csv`.
# Every row of that file is below. The step time is the trainer's own last
# timing line. The memory is a peak of a one-second sampler over this arm's OWN
# processes, so the 4.0 GB of other work on the card is not in it.
#
#   arm                step time   memory      40,000 steps
#   k3_r100_09_dec       179 ms     6,436 MiB     2.0 h
#   k3_r100_09b          266 ms     6,472 MiB     3.0 h
#   k3_r100_09           272 ms     6,436 MiB     3.0 h
#   k8_r100_09           346 ms     7,160 MiB     3.8 h
#   k32_r100_09_dec      550 ms    10,058 MiB     6.1 h
#   k32_r200_08          651 ms    10,062 MiB     7.2 h
#   k32_r100_09          688 ms    10,062 MiB     7.6 h
#
# THE TWO BRACKET ARMS HAVE NO ROW. They are configuration 1 at another
# learning rate, and the step time and the memory come from the depth, the
# reduction and the decay. So each one reads the 272 ms and the 6,436 MiB of
# `k3_r100_09`.
#
# A DECAY ARM READS FASTER because its `L_rep` weight is 0.0 over most of a
# smoke. The study ends that ramp at step 2,000, so 38,000 of its 40,000 steps
# run at this lower cost.
#
# THE STEP TIME IS AN UPPER BOUND. The card carried 4.0 GB and 80 to 97 percent
# of other work through every arm. The six mandatory arms add up to 28.9 h of
# backbone at these readings, and the optional `k8_r100_09` adds 3.8 h.
#
# TWO k = 32 ARMS DO NOT SHARE ONE CARD HERE. 10.1 + 10.1 + 4.0 = 24.2 GB on
# a 24.0 GB card. Card A runs its two k = 32 arms one after the other, which
# is what the order below does.
#
# ---- The run plan, from PR #413 ----------------------------------------------
#
# STOP AT 40,000 STEPS FIRST. Do not launch 200,000 steps for every arm: that
# is 164 GPU-hours of backbone where the plan below is 28.9. One k = 32 arm
# that climbs to 200,000 steps adds 30.6. `phase1.sh` defaults to this pass.
#
#   Card A:  BB_GPU=0 STOPS=40000 ARMS="k32_r100_09 k32_r100_09_dec" \
#              bash run.sh phase1
#   Card B:  BB_GPU=1 STOPS=40000 \
#              ARMS="k3_r100_09 k3_r100_09b k3_r100_09_dec k32_r200_08" \
#              bash run.sh phase1
#
# `k8_r100_09` is optional. It goes last, and only on an idle card. This order
# completes the decay pair and the seed band first, so an interruption still
# leaves an answer. Cost 28.9 GPU-hours of backbone.
#
# Seeds: 20260520 on every arm, 20260525 on `k3_r100_09b` alone. No new seed.
#
# ---- The learning-rate bracket -----------------------------------------------
#
# Every arm above trains at 1e-3. That rate is the Moirai recipe. This project
# does not use muP, and the trainer builds ONE AdamW group over all parameters
# with no width multiplier, so a rate that fits `d_model` 64 need not fit 384.
# A capacity verdict taken at a rate that does not fit the width answers a
# question about the rate.
#
# Two arms bracket the rate on the cheapest cell, configuration 1 at seed
# 20260520. Each one moves the RATE column of `arms.tsv` and no other.
#
#   k3_r100_09_lr33   3.3e-4
#   k3_r100_09_lr17   1.67e-4, which is 1e-3 times 64/384
#
# THEY GO ON CARD B, after the `k3_r100_09b` head. Cost about 13 GPU-hours with
# their heads and evals. Do not launch them before that head ends.
#
#   Card B:  BB_GPU=1 STOPS=40000 \
#              ARMS="k3_r100_09_lr33 k3_r100_09_lr17" bash run.sh phase1
#
# THE RULE. D = 1.3495 minus the better rate arm, where 1.3495 is
# `k3_r100_09` at 40,000 steps and 1e-3.
#
#   * D above 0.0471, the band: the rate does not fit the width. Every 1e-3
#     number of this card is void, and phase 1 re-runs at the winning rate.
#   * Both arms inside the band, or worse: the learning rate does not explain
#     the 11.4M deficit at width 384.
#
# A slower arm reaches the contrastive task later, so `cf412_auc_warmup` scales
# the AUC warm-up by the rate ratio. Without that scale the gate can stop a
# healthy slow arm on the rows of its first minutes.
#
# ---- The gate, then 200,000 steps --------------------------------------------
#
# THE BAND is 0.0471 GM-Relative MASE, or the spread of the two `k3_r100_09`
# seeds, whichever is wider. #409 measured 0.0471 at this stop.
#
#   * THE AUC COMES FIRST. An arm whose rolling AUC median ended under 0.55
#     lost the contrastive task, and it does not climb. This applies to
#     every arm, `k3_r100_09` included, because a higher stop trains the
#     same collapse. The gate writes `results/collapsed_<arm>.txt`, which
#     names the step, and `phase1.sh` reads those files before its first
#     leg. Delete that file to let the arm run again.
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
# Usage:  BB_GPU=0 bash run.sh              # every stage, phase1 at 40,000 steps
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
