#!/bin/bash
# #393 — keep the bb40k replicate pool fed until every tier is scored.
#
# Usage: WT=<checkout> RUNS=<durable root> nohup bash scripts/bb40k_supervisor.sh &
#
# `seed_replicates_bb40k.sh` walks its job list once and exits when the list
# is drained. That is the right shape for one pass and the wrong shape for a
# night: a job whose eval died releases its claim and needs another pass,
# and the three tiers are a queue rather than one batch — t1 has to finish
# before there is any reason to spend a GPU on t2.
#
# So this waits for the running pass to exit, then starts the next with the
# full tier list, and keeps going until the score files say there is nothing
# left. Passes are bounded: a job that fails the same way on every pass is a
# real failure and should stop costing GPU time rather than loop forever.
#
# It is idempotent and safe to start twice: the pool's claims are `mkdir`,
# so a second supervisor's driver takes only jobs nobody holds.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
LOG="$RES/bb40k_supervisor.log"

export WT="${WT:-$(dirname "$(dirname "$EXP")")}"
export RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
# t1 only — see the CANCELLED note in seed_replicates_bb40k.sh. This default
# used to be `t1,hw,t2`, which is how a supervisor launched without an explicit
# tier list would have queued the cancelled tiers on its next pass.
TIERS="${CF393_BB40K_TIERS:-t1}"
JOBS="${CF393_BB40K_JOBS:-6}"
MAX_PASSES="${CF393_BB40K_PASSES:-6}"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] [super] $*" | tee -a "$LOG"; }

remaining(){
  CF393_BB40K_TIERS="$TIERS" bash "$HERE/seed_replicates_bb40k.sh" --status \
    | awk '/scored,/{split($1,a,"/"); print a[2] - a[1]}'
}

# Wait out whatever pass is already running, so this does not widen the pool
# past what the GPUs were sized for.
while pgrep -f 'seed_replicates_bb40k.sh$' >/dev/null 2>&1; do sleep 60; done

for (( pass = 1; pass <= MAX_PASSES; pass++ )); do
  left="$(remaining)"
  if [ "${left:-1}" -le 0 ]; then
    say "all scored after $(( pass - 1 )) pass(es)"
    break
  fi
  say "pass $pass/$MAX_PASSES — $left job(s) left, tiers=$TIERS, pool=$JOBS"
  CF393_BB40K_TIERS="$TIERS" CF393_BB40K_JOBS="$JOBS" \
    bash "$HERE/seed_replicates_bb40k.sh" >>"$RES/seed_bb40k_driver.log" 2>&1
  # A pass that ends with the same count as it started made no progress, so
  # the next one would not either.
  after="$(remaining)"
  say "pass $pass done — $after job(s) left"
  if [ "${after:-0}" -eq "${left:-0}" ] && [ "${after:-0}" -gt 0 ]; then
    say "no progress in pass $pass; stopping. See results/bb40k_*.log"
    break
  fi
done

say "supervisor exiting — $(remaining) job(s) unscored"
python3 "$HERE/paired_delta.py" >>"$RES/paired_delta.log" 2>&1
say "paired_delta.py refreshed"
