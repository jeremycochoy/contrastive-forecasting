#!/bin/bash
# #401 phase 1 — the three arms, their stops, their heads and their evals.
#
# Order, and why it is this order. The card runs k = 16 first, then k = 8,
# then k = 32: k = 16 answers the question, and the other two bracket it, so
# a session that runs out of GPU time still holds an answer. Within an arm
# the stops climb, because a leg resumes the one below it.
#
# Each stop's head starts as soon as its checkpoint lands, while the backbone
# climbs to the next stop. Head training and backbone training both want the
# card, so `head_eval_bb.sh` waits for free VRAM (its `head_vram_gate`) and
# the GIFT-Eval that follows runs on the CPU. Set HEAD_BG=0 to run each head
# inline instead, which is slower and easier to read.
#
# Usage:  bash phase1.sh                       # all three arms
#         DEPTHS=16 bash phase1.sh             # one arm
#         CF401_DRY_RUN=1 bash phase1.sh       # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

DEPTHS="${DEPTHS:-$CF401_DEPTHS}"
HEAD_BG="${HEAD_BG:-1}"
BB_GPU="${BB_GPU:-0}"
HEAD_GPU="${HEAD_GPU:-$BB_GPU}"
mkdir -p "$CF401_RESULTS"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 phase1] $*" \
  | tee -a "$CF401_RESULTS/phase1.log"; }

heads=(); head_names=(); inline_failed=0
for k in $DEPTHS; do
  cf401_require_depth "$k" || exit $?
  for stop in $CF401_STOPS; do
    if [ -n "${CF401_DRY_RUN:-}" ]; then
      echo "arm  k=$k steps=$stop"
      echo "head k=$k stop=$stop steps=$CF401_HEAD_STEPS_P1 enc=$CF401_ENC"
      continue
    fi

    log "arm k=$k -> $stop"
    BB_GPU="$BB_GPU" bash "$HERE/run_arm_k.sh" "$k" "$stop"
    rc=$?
    if [ $rc -ne 0 ]; then
      # The next stop resumes this one, so a failed leg makes every stop
      # above it meaningless. Stop this arm and start the next.
      log "arm k=$k stop=$stop rc=$rc — skipping the rest of this arm"
      break
    fi

    if [ "$HEAD_BG" = "1" ]; then
      log "head k=$k stop=$stop (background)"
      # `nohup`, not `nohup setsid`. setsid forks when the caller already
      # leads its process group, and then `$!` is the PID of a process that
      # exits at once — so the `wait` below would return before the head had
      # started. nohup execs in place, so `$!` is the head.
      BB_GPU="$HEAD_GPU" nohup bash "$HERE/head_eval.sh" "$k" "$stop" \
        >>"$CF401_RESULTS/head_k${k}_bb$(cf401_steps_label "$stop").out" 2>&1 &
      heads+=($!); head_names+=("k=$k stop=$stop")
    else
      log "head k=$k stop=$stop (inline)"
      BB_GPU="$HEAD_GPU" bash "$HERE/head_eval.sh" "$k" "$stop"
      rc=$?
      log "head k=$k stop=$stop rc=$rc"
      [ $rc -eq 0 ] || inline_failed=$(( inline_failed + 1 ))
    fi
  done
done

[ -n "${CF401_DRY_RUN:-}" ] && exit 0

# A head that is still running when the last backbone finishes is the normal
# case, and its GIFT-Eval is hours of CPU. Waiting here is what makes
# `collect.sh` afterwards see every score.
#
# Every rc is logged, named by its (k, stop). A discarded `wait` left a dead
# head with no line in phase1.log, and the failure then surfaced hours later
# as an "incomplete phase 1" abort from the picker.
failed="$inline_failed"
if [ "${#heads[@]}" -gt 0 ]; then
  log "waiting for ${#heads[@]} head+eval job(s)"
  for i in "${!heads[@]}"; do
    wait "${heads[$i]}"; rc=$?
    if [ $rc -eq 0 ]; then
      log "head ${head_names[$i]} rc=0"
    else
      failed=$(( failed + 1 ))
      log "head ${head_names[$i]} rc=$rc — see head_k*.out in $CF401_RESULTS"
    fi
  done
fi

bash "$HERE/collect.sh"
log "phase 1 drained — $(wc -l <"$CF401_RESULTS/scores.csv") row(s) in scores.csv, $failed head(s) failed"
[ "$failed" -eq 0 ] || exit 1
