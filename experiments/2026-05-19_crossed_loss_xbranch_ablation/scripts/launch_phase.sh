#!/bin/bash
# Start a box_run phase for <arm> on a box (default: that arm's own box
# from state/<arm>.env; override with HOST_ARM to run on another arm's
# box, e.g. packing 2 q-heads on one 2-GPU box). Backbone for the
# overridden arm must already be present on the target box.
#   launch_phase.sh <arm> <phase> [gpu]            (on arm's own box)
#   HOST_ARM=<other> launch_phase.sh <arm> <phase> [gpu]   (on other's box)
set -uo pipefail
EXP=/home/jupyter/cf-wt-crossed-loss/experiments/2026-05-19_crossed_loss_xbranch_ablation
source "$EXP/scripts/_ssh.sh"
ARM="${1:?arm}"; PHASE="${2:?phase}"; GPU="${3:-0}"
HENV="$EXP/scripts/state/${HOST_ARM:-$ARM}.env"
[ -f "$HENV" ] || { echo "no state $HENV"; exit 1; }
. "$HENV"
read -r H P < <(ssh_coords "$INST") || { H="$HOST"; P="$PORT"; }
RL="/workspace/app/box_${ARM}_${PHASE}_g${GPU}.log"
echo "[$ARM] launch $PHASE gpu=$GPU on inst $INST ($H:$P) host_arm=${HOST_ARM:-$ARM}"
ssh $SSHO -p "$P" "root@$H" \
  "cd /workspace/app && setsid bash -c 'bash experiments/2026-05-19_crossed_loss_xbranch_ablation/scripts/box_run.sh $PHASE $ARM $GPU > $RL 2>&1' < /dev/null & echo started \$!"
sleep 3
ssh $SSHO -p "$P" "root@$H" "tail -4 $RL 2>/dev/null"
