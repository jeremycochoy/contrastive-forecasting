#!/bin/bash
# Condition 3 of the PR #394 decision, enforced by the scheduler rather than
# by hand: a box is given back once its ladder driver has exited and its
# work is home. release_idle_boxes.sh holds the five gates; this only paces
# it. 15-min passes, and the script itself needs two consecutive idle reads,
# so a box is confirmed idle for 30 minutes before anything is destroyed.
#
# DRAIN=1 since the budget correction of 19:00 on 08-05. `vastrun-status`
# reports spend per RUNNING instance, so a released box's spend leaves the
# total and summing it understates what the account has paid. The real
# figure comes from `vastrun-balance`, and it was $34.93 against an $80.26
# envelope — about 12 h of fleet time, not the 17 h the summed figure
# suggested. Under that, a box waiting on an eval elsewhere is the main
# leak: see the DRAIN gate in release_idle_boxes.sh.
cd /tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder || exit 2
while :; do
  DRAIN=1 RELEASE=1 bash scripts/release_idle_boxes.sh
  sleep 900
done
