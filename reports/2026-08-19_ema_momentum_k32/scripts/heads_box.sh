#!/bin/bash
# #404 — the four student heads on the box, one per card, at the same time.
#
# `heads_watch.sh` is elisa's watcher: it fires one head at a time, on the
# checkpoints the sync loop lands, because elisa has one part-free lane. The
# box has four idle 5090s and the four backbones are already on its own disk,
# so the four heads run there together and the box is released hours earlier.
#
# The box carries no `gift_eval` package and no gift-eval data, so
# `head_eval_bb.sh` trains the head, saves it, and then stops at the eval with
# `ABORT: no eval script`. That is the intended end here. It writes no score
# file, so elisa's `collect.sh` cannot read a half-made result. elisa re-fires
# the same script per arm, which skips the head that is already on disk and
# runs only its 97-config GIFT-Eval.
#
# So a non-zero return from an arm is NOT a failure by itself. This script
# judges each arm on its head checkpoint, by name, and prints the size.
#
# Usage, on the box:
#   cd /root/cf/reports/2026-08-19_ema_momentum_k32
#   nohup setsid bash scripts/heads_box.sh &
#
#   GPUS="0 1" bash scripts/heads_box.sh         # which cards to use
#   CF404_DRY_RUN=1 bash scripts/heads_box.sh    # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_BOX_RUNS"

# The cards this box carries, not a fixed four. See launch_box.sh.
GPUS="${GPUS:-$(cf404_default_gpus)}"
ARMS="${ARMS:-$CF404_ARMS}"
STOP="${STOP:-$CF404_STOPS}"
STAGGER="${STAGGER:-60}"
mkdir -p "$CF404_RESULTS"

LOG="$CF404_RESULTS/heads_box.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 heads box] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$GPUS"
read -r -a arm_list <<<"$ARMS"
[ "${#gpu_list[@]}" -ge 1 ] || { echo "ABORT: GPUS is empty" >&2; exit 2; }
# The same card check the backbone launcher runs. A head lane on a card the
# box does not carry dies the same way.
cf404_require_gpus "$GPUS" || exit 2
cf404_require_stop "$STOP" || exit $?
for arm in "${arm_list[@]}"; do cf404_require_arm "$arm" || exit $?; done

# The head checkpoint `head_eval_bb.sh` writes for one arm. Its name is that
# script's own: `qhead_<tag>_s<seed>_final.pth` under the tag's eval dir.
head_ckpt(){  # <arm>
  local tag
  tag="$(cf404_tag "$1" "$STOP" "$CF404_HEAD_STEPS")"
  printf '%s/qhead_%s_s%s_final.pth\n' "$(cf404_eval_dir "$1" "$tag")" \
    "$tag" "${HEAD_SEED:-20260722}"
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "heads box root=$CF404_ROOT results=$CF404_RESULTS gpus='$GPUS'"
  echo "  budget=$CF404_HEAD_STEPS enc=$CF404_ENC stop=$STOP"
  for i in "${!arm_list[@]}"; do
    echo "head ${arm_list[$i]} gpu=${gpu_list[$(( i % ${#gpu_list[@]} ))]}" \
         "bb=$(cf404_bb_ckpt "${arm_list[$i]}" "$STOP" || true)"
    echo "  head=$(head_ckpt "${arm_list[$i]}")"
  done
  exit 0
fi

log "START arms='$ARMS' gpus='$GPUS' budget=$CF404_HEAD_STEPS root=$CF404_ROOT"
pids=(); names=()
for i in "${!arm_list[@]}"; do
  arm="${arm_list[$i]}"
  gpu="${gpu_list[$(( i % ${#gpu_list[@]} ))]}"
  if [ -f "$(head_ckpt "$arm")" ]; then
    log "head $arm SKIP — already on disk"
    continue
  fi
  log "head $arm on gpu $gpu"
  BB_GPU="$gpu" nohup bash "$HERE/head_eval.sh" "$arm" "$STOP" \
    >>"$CF404_RESULTS/head_${arm}_bb$(cf404_steps_label "$STOP").out" 2>&1 &
  pids+=($!); names+=("$arm gpu=$gpu")
  # The head trainer streams the same HuggingFace dataset on every lane.
  # Starting four cold readers together puts them on one connection.
  [ "$i" -lt $(( ${#arm_list[@]} - 1 )) ] && sleep "$STAGGER"
done

for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  log "arm ${names[$i]} rc=$rc (rc!=0 is expected: this box has no eval)"
done

# The verdict is the checkpoint, not the return code.
missing=0
for arm in "${arm_list[@]}"; do
  ck="$(head_ckpt "$arm")"
  if [ -f "$ck" ]; then
    log "head $arm OK $(wc -c <"$ck") B $(basename "$ck")"
  else
    log "head $arm MISSING $ck"
    missing=$(( missing + 1 ))
  fi
done
log "HEADS DONE — $missing of ${#arm_list[@]} head(s) missing"
[ "$missing" -eq 0 ] || exit 1
