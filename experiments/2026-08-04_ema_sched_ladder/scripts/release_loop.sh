#!/bin/bash
# Condition 3 of the PR #394 decision, enforced by the scheduler rather than
# by hand: a box is given back once its ladder driver has exited and its
# work is home. release_idle_boxes.sh holds the five gates; this only paces
# it. 15-min passes, and the script itself needs two consecutive idle reads,
# so a box is confirmed idle for 30 minutes before anything is destroyed.
cd /tmp/contrastive-forecasting-393/experiments/2026-08-04_ema_sched_ladder || exit 2
while :; do
  RELEASE=1 bash scripts/release_idle_boxes.sh
  sleep 900
done
