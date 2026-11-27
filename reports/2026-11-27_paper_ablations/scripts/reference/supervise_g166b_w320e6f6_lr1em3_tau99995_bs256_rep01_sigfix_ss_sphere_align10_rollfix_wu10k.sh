#!/bin/bash
# Keeps g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k alive: it starts run_g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k.sh again (which resumes from the last checkpoint) when the trainer
# ends before the last step. It stops at the last checkpoint, at the file $R/stop, or after MAX_STARTS starts.
# Written by /workspace/hp64/make_run.py (the supervisor of the 166k runs).
#   cd /workspace/hp64/g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k && BB_GPU=<gpu> setsid nohup bash supervise_g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k.sh >> logs/supervise.log 2>&1 < /dev/null &
set -uo pipefail
R="${R:-/workspace/hp64/g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k}"
MAX_STARTS="${MAX_STARTS:-5}"
LAST_K="${LAST_K:-166}"
for start in $(seq 1 "$MAX_STARTS"); do
  if [ -n "$(ls "$R"/ckpt/g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k_"${LAST_K}"k.pth "$R"/ckpt/g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k_r[0-9]*_"${LAST_K}"k.pth 2>/dev/null)" ]; then
    echo "$(date -u +%FT%TZ) supervise: the ${LAST_K}k checkpoint exists, end"; exit 0
  fi
  if [ -e "$R/stop" ]; then
    echo "$(date -u +%FT%TZ) supervise: stop file, end"; exit 0
  fi
  echo "$(date -u +%FT%TZ) supervise: start $start of $MAX_STARTS"
  bash "$R/run_g166b_w320e6f6_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_wu10k.sh" >>"$R/logs/train.log" 2>&1
  echo "$(date -u +%FT%TZ) supervise: the trainer ended with code $?"
  sleep 120
done
echo "$(date -u +%FT%TZ) supervise: $MAX_STARTS starts used, end"; exit 1
