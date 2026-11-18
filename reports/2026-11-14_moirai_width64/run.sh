#!/bin/bash
# #462: MPE (our copy of Moirai, #421ew) at the size of the width-64 runs:
# width 64, 1 encoder layer, 3 forecaster layers, 166,000 steps at batch 256.
# The run outputs and the logs go under $R (CF462_ROOT, default
# /home/jupyter/cf_runs/cf462), because a reboot of elisa deletes /tmp.
#
#   smoke   SMOKE_STEPS (300) steps in $R/smoke, the A2V score of their save
#           on 2 configs, and the loss figure of these steps
#   train   the run to the 8 stops of MPE, with restarts (scripts/train.sh)
#   score   the GM-Relative MASE of each saved stop with the value heads of
#           the model (A2V, scripts/score_stops.sh). FOLLOW=1 also waits
#           for the next stops while the run trains.
#   plots   the width-64 GM-Relative MASE graph and the loss figure
#
# Usage, from the checkout:
#   R=/home/jupyter/cf_runs/cf462; mkdir -p $R/logs
#   setsid nohup bash reports/2026-11-14_moirai_width64/run.sh train \
#     >> $R/logs/supervise.log 2>&1 < /dev/null &
#   FOLLOW=1 setsid nohup bash reports/2026-11-14_moirai_width64/run.sh score \
#     >> $R/logs/score.log 2>&1 < /dev/null &
#   bash reports/2026-11-14_moirai_width64/run.sh plots
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="${CF462_ROOT:-/home/jupyter/cf_runs/cf462}"
STAGES="${*:-smoke}"
SMOKE_STEPS="${SMOKE_STEPS:-300}"

smoke() {
  SMOKE_STEPS="$SMOKE_STEPS" MAX_STARTS=1 bash "$HERE/scripts/train.sh" || return
  CF462_ROOT="$R/smoke" RESULTS="$R/smoke/results" STOPS=$((SMOKE_STEPS / 1000)) \
    EVAL_CONFIG_FILTER='^(ett2/D/short|bizitobs_l2c/H/short)$' EVAL_EXPECT_CONFIGS=2 \
    bash "$HERE/scripts/score_stops.sh" || return
  LOSS_FIRST_STEP=10 python3 "$HERE/scripts/plot_loss_composition.py" "$R/smoke" \
    "$R/smoke/plots/loss_composition.png" "$R/smoke/results/loss_parts_binned.tsv"
}

plots() {
  python3 "$HERE/scripts/plot_width64_gm_mase.py" &&
    python3 "$HERE/scripts/plot_loss_composition.py" "$R"
}

for stage in $STAGES; do
  case "$stage" in
    smoke) smoke ;;
    train) bash "$HERE/scripts/train.sh" ;;
    score) bash "$HERE/scripts/score_stops.sh" ;;
    plots) plots ;;
    *) echo "ABORT: unknown stage '$stage' (smoke train score plots)" >&2; exit 2 ;;
  esac
  rc=$?
  [ $rc -eq 0 ] || { echo "ABORT: stage '$stage' rc=$rc" >&2; exit $rc; }
done
