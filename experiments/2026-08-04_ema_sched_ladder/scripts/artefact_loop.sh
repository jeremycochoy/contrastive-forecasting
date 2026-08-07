#!/bin/bash
# #393 — pull every machine's artefacts into the branch as they land.
#
# Usage:  nohup setsid bash scripts/artefact_loop.sh &
#
# The brief asks for artefacts committed as they land. Each machine keeps
# its own ladder.csv and its own evidence files, so "as they land" means
# collect_results.sh first and a commit second. Commits only when something
# changed, so the branch does not fill with empty commits overnight.
#
# Checkpoints are not collected — `*.pth` is gitignored, they are 5 MB each,
# and they already live on the durable roots and in the six sync trees.
set -uo pipefail
WT="${WT:-/tmp/contrastive-forecasting-393}"
EXP="$WT/experiments/2026-08-04_ema_sched_ladder"
INTERVAL="${ARTEFACT_INTERVAL:-1800}"
cd "$WT" || exit 2

while :; do
  WT="$WT" bash "$EXP/scripts/collect_results.sh" >> "$EXP/results/collect.log" 2>&1
  git add -A experiments/2026-08-04_ema_sched_ladder 2>/dev/null
  if ! git diff --cached --quiet 2>/dev/null; then
    n_scores=$(( $(wc -l < "$EXP/results/ladder_all.csv" 2>/dev/null || echo 1) - 1 ))
    git -c user.name="Jeremy Cochoy" -c user.email="jeremy@redstone.ee" \
      commit -q -m "ema_sched_ladder: artefacts at $(date -u '+%H:%M')Z, ${n_scores} scored stop(s)" \
      && git push -q origin feature/contrastive-forecasting-393 2>/dev/null
  else
    git reset -q
  fi
  sleep "$INTERVAL"
done
