#!/bin/bash
# #404 — the four backbone arms on one rented box, two GPUs.
#
# The box trains BACKBONES ONLY. It carries the trainer, the HF token and
# nothing else this study needs: the 97-config GIFT-Eval reads gift-eval-data
# and the `gift_eval` package, both of which live on elisa. A head trained here
# would still wait for elisa to score it, so the heads run there, on the
# checkpoints the sync loop pulls (`scripts/heads_watch.sh`).
#
# Two cards, four arms, so each card takes two arms in turn. The arms are
# independent — no arm resumes another — so the split is free. At the step time
# #401 measured for k = 32 under this objective, one arm to 40,000 steps is
# about 5.4 hours, and the box finishes in about 11 hours.
#
# Everything is idempotent. A box that reboots loses the legs in flight and
# nothing else: re-run this script and every finished arm is a no-op, because a
# leg resumes the cell's furthest checkpoint with its optimizer state.
#
# The root is CF404_BOX_RUNS, on the box's own disk. It is not a knob to
# remember: the sync loop pulls THAT path, so a box that saved anywhere else
# would climb for 11 hours and never reach elisa.
#
# Usage, on the box, after the checkout and the HF token are in place:
#   cd /root/cf/reports/2026-08-19_ema_momentum_k32
#   nohup setsid bash scripts/launch_box.sh &
#
#   GPUS="0 1" bash scripts/launch_box.sh        # which cards to use
#                                                # (default: every card here)
#   CF404_DRY_RUN=1 bash scripts/launch_box.sh   # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_BOX_RUNS"

# The cards this box carries, not a fixed pair. Round 3's plan print read
# `gpu=1` on a one-card box because this default was `0 1`.
GPUS="${GPUS:-$(cf404_default_gpus)}"
ARMS="${ARMS:-$CF404_ARMS}"
STAGGER="${STAGGER:-180}"
mkdir -p "$CF404_RESULTS"

LOG="$CF404_RESULTS/launch_box.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 box] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$GPUS"
read -r -a arm_list <<<"$ARMS"
[ "${#gpu_list[@]}" -ge 1 ] || { echo "ABORT: GPUS is empty" >&2; exit 2; }
# Every index in GPUS must be a card this box carries. The check runs before
# the plan print as well, so a plan that names a card that is not there fails
# here and not five hours later inside `.to(device)`.
cf404_require_gpus "$GPUS" || exit 2
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done
# The box is rented by the hour, so the checkout it carries is checked before
# the first leg and not after eleven of them. See cf404_check_checkout.
cf404_check_checkout || exit 6

# Arms are dealt round-robin over the cards, so two cards take two arms each
# and both finish in two passes.
lane_of(){  # <arm index>
  echo $(( $1 % ${#gpu_list[@]} ))
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "box reduce=$CF404_REDUCE root=$CF404_ROOT results=$CF404_RESULTS"
  for i in "${!arm_list[@]}"; do
    echo "arm ${arm_list[$i]} gpu=${gpu_list[$(lane_of "$i")]} stops='$CF404_STOPS' heads=0"
  done
  exit 0
fi

# `results/MACHINE` names this box for #373's cell-claim check. A vast.ai
# container's `hostname` is a per-boot id, so it cannot stand in for a name a
# committed file is written against.
[ -f "$CF404_RESULTS/MACHINE" ] || echo "${CF404_BOX_NAME:-box}" >"$CF404_RESULTS/MACHINE"

log "START arms='$ARMS' gpus='$GPUS' root=$CF404_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  2>/dev/null | sed 's/^/  gpu /' | tee -a "$LOG"

pids=(); names=()
for lane in "${!gpu_list[@]}"; do
  gpu="${gpu_list[$lane]}"
  lane_arms=""
  for i in "${!arm_list[@]}"; do
    [ "$(lane_of "$i")" = "$lane" ] && lane_arms="$lane_arms ${arm_list[$i]}"
  done
  [ -n "$lane_arms" ] || continue
  log "gpu $gpu takes${lane_arms}"
  ARMS="${lane_arms# }" BB_GPU="$gpu" CF404_HEADS=0 \
    nohup bash "$HERE/phase1.sh" \
      >>"$CF404_RESULTS/phase1_gpu${gpu}.out" 2>&1 &
  pids+=($!); names+=("gpu=$gpu arms=${lane_arms# }")
  # Every lane opens the same streaming dataset. Starting them together puts
  # two cold HF readers on one connection.
  sleep "$STAGGER"
done

failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  log "lane ${names[$i]} rc=$rc"
  [ $rc -eq 0 ] || failed=$(( failed + 1 ))
done
log "BOX DONE — $failed lane(s) failed"
[ "$failed" -eq 0 ] || exit 1
