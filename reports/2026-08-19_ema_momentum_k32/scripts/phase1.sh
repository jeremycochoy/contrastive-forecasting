#!/bin/bash
# #404 — the four arms, their heads and their evals, in the order the card
# runs them.
#
# One arm is one backbone to 40,000 steps, then one 30,000-step student head,
# then that head's 97 GIFT-Eval configs. The arms are independent: no arm
# resumes another, so the order between them is free and two GPUs run two arms
# at a time (`launch_box.sh`).
#
# Each arm's head starts as soon as its checkpoint lands, while the next arm
# trains. Head training and backbone training both want the card, so
# `head_eval_bb.sh` waits for free VRAM (its `head_vram_gate`) and the
# GIFT-Eval that follows runs on the CPU. Set HEAD_BG=0 to run each head
# inline instead, which is slower and easier to read.
#
# CF404_HEADS=0 trains the backbones and nothing else. That is what a rented
# box runs: no GIFT-Eval data and no gift_eval package live there. elisa runs
# `heads_watch.sh` instead, which fires each head as its checkpoint lands
# through the sync loop.
#
# Usage:  bash phase1.sh                        # every arm, heads included
#         ARMS="a08 a09" bash phase1.sh         # two arms
#         CF404_HEADS=0 ARMS=a08 bash phase1.sh # backbone only, on a box
#         CF404_DRY_RUN=1 bash phase1.sh        # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARMS="${ARMS:-$CF404_ARMS}"
HEAD_BG="${HEAD_BG:-1}"
HEADS="${CF404_HEADS:-1}"
BB_GPU="${BB_GPU:-0}"
HEAD_GPU="${HEAD_GPU:-$BB_GPU}"
mkdir -p "$CF404_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 phase1] $*" \
  | tee -a "$CF404_RESULTS/phase1.log"; }

heads=(); head_names=(); inline_failed=0; legs_failed=0
for arm in $ARMS; do
  cf404_require_arm "$arm" || exit $?
  for stop in $CF404_STOPS; do
    if [ -n "${CF404_DRY_RUN:-}" ]; then
      echo "arm $arm steps=$stop ema=$(cf404_ema_args "$arm")"
      [ "$HEADS" = "1" ] && \
        echo "head $arm stop=$stop steps=$CF404_HEAD_STEPS enc=$CF404_ENC"
      continue
    fi

    log "arm $arm -> $stop"
    BB_GPU="$BB_GPU" bash "$HERE/run_arm.sh" "$arm" "$stop"
    rc=$?
    if [ $rc -ne 0 ]; then
      # The count is carried to the exit status. A phase that reported success
      # here would leave a dead arm to surface hours later, as a table with
      # three rows and no reason for the fourth.
      legs_failed=$(( legs_failed + 1 ))
      log "arm $arm stop=$stop rc=$rc — no head for this arm"
      continue
    fi

    if [ "$HEADS" != "1" ]; then
      log "head $arm SKIPPED (CF404_HEADS=0, this box trains backbones)"
    elif [ "$HEAD_BG" = "1" ]; then
      log "head $arm stop=$stop (background)"
      # `nohup`, not `nohup setsid`. setsid forks when the caller already leads
      # its process group, and then `$!` is the PID of a process that exits at
      # once — so the `wait` below would return before the head had started.
      BB_GPU="$HEAD_GPU" nohup bash "$HERE/head_eval.sh" "$arm" "$stop" \
        >>"$CF404_RESULTS/head_${arm}_bb$(cf404_steps_label "$stop").out" 2>&1 &
      heads+=($!); head_names+=("$arm stop=$stop")
    else
      log "head $arm stop=$stop (inline)"
      BB_GPU="$HEAD_GPU" bash "$HERE/head_eval.sh" "$arm" "$stop"
      rc=$?
      log "head $arm stop=$stop rc=$rc"
      [ $rc -eq 0 ] || inline_failed=$(( inline_failed + 1 ))
    fi
  done
done

[ -n "${CF404_DRY_RUN:-}" ] && exit 0

# A head still running when the last backbone finishes is the normal case, and
# its GIFT-Eval is hours of CPU. Waiting here is what makes `collect.sh`
# afterwards see every score.
failed="$inline_failed"
if [ "${#heads[@]}" -gt 0 ]; then
  log "waiting for ${#heads[@]} head+eval job(s)"
  for i in "${!heads[@]}"; do
    wait "${heads[$i]}"; rc=$?
    if [ $rc -eq 0 ]; then
      log "head ${head_names[$i]} rc=0"
    else
      failed=$(( failed + 1 ))
      log "head ${head_names[$i]} rc=$rc — see head_*.out in $CF404_RESULTS"
    fi
  done
fi

# A backbone-only run scores nothing, so there is nothing to collect. Running
# collect.sh there would write an empty scores.csv over the one the machine
# that DOES score keeps.
if [ "$HEADS" = "1" ]; then
  bash "$HERE/collect.sh"
  log "phase 1 drained — $legs_failed leg(s) and $failed head(s) failed"
else
  log "phase 1 backbones drained — $legs_failed leg(s) failed, heads run elsewhere"
fi
failed=$(( failed + legs_failed ))
[ "$failed" -eq 0 ] || exit 1
