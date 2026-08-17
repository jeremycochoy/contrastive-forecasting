#!/bin/bash
# #401 — the two backbone arms on one rented box, one arm per GPU.
#
# The box trains BACKBONES ONLY. It carries the trainer, the HF token and
# nothing else that this study needs: the 97-config GIFT-Eval reads
# gift-eval-data and the gift_eval package, both of which live on elisa. A
# head trained here would still wait for elisa to score it, so the heads run
# on elisa, on the checkpoints the sync loop pulls (`scripts/heads_watch.sh`).
#
# One GPU per arm, both at once. k = 8 costs about 14 h to 200k steps and
# k = 32 about 30 h, at the step times measured on an RTX 4090
# (results/smoke_k16.csv). So the box finishes phase 1 in the time the
# k = 32 arm takes, and the k = 8 card frees for anything queued after it.
#
# Peak memory is 5.9 GiB per leg, measured, and it does not grow with the
# depth. Two legs therefore fit one 24 GiB card, but they do not share one
# here: two cards are free and one leg per card is faster.
#
# Everything is idempotent. A box that reboots loses the legs in flight and
# nothing else: re-run this script and every finished stop is a no-op,
# because a leg resumes the cell's furthest checkpoint with its optimizer
# state.
#
# Usage, on the box, after bootstrap_box.sh:
#   cd /root/cf/reports/2026-08-15_rollout_depth_k16_8_32
#   CF401_ROOT=/root/cf401_runs nohup setsid bash scripts/launch_box.sh &
#
#   GPUS="0 1" bash scripts/launch_box.sh        # which cards to use
#   CF401_DRY_RUN=1 bash scripts/launch_box.sh   # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

# One card per arm, paired by position: the first depth takes the first GPU.
# Both cards are free at once, so the order does not decide the wall clock.
GPUS="${GPUS:-0 1}"
DEPTHS="${DEPTHS:-$CF401_DEPTHS}"
STAGGER="${STAGGER:-180}"
mkdir -p "$CF401_RESULTS"

LOG="$CF401_RESULTS/launch_box.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 box] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$GPUS"
read -r -a depth_list <<<"$DEPTHS"
[ "${#gpu_list[@]}" -ge "${#depth_list[@]}" ] || {
  echo "ABORT: ${#depth_list[@]} arm(s) and only ${#gpu_list[@]} GPU(s)." >&2
  echo "  Set GPUS to one card per arm, or DEPTHS to fewer arms." >&2
  exit 2; }
for k in "${depth_list[@]}"; do cf401_require_depth "$k" || exit $?; done

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "box reduce=$CF401_REDUCE root=$CF401_ROOT results=$CF401_RESULTS"
  for i in "${!depth_list[@]}"; do
    echo "arm k=${depth_list[$i]} gpu=${gpu_list[$i]} stops='$CF401_STOPS' heads=0"
  done
  exit 0
fi

# `results/MACHINE` names this box for #373's cell-claim check. A vast.ai
# container's `hostname` is a per-boot id, so it cannot stand in for a name a
# committed file is written against.
[ -f "$CF401_RESULTS/MACHINE" ] || echo "${CF401_BOX_NAME:-box}" >"$CF401_RESULTS/MACHINE"

log "START reduce=$CF401_REDUCE depths='$DEPTHS' gpus='$GPUS' root=$CF401_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  2>/dev/null | sed 's/^/  gpu /' | tee -a "$LOG"

pids=(); names=()
for i in "${!depth_list[@]}"; do
  k="${depth_list[$i]}"; gpu="${gpu_list[$i]}"
  log "arm k=$k START on gpu $gpu"
  DEPTHS="$k" BB_GPU="$gpu" CF401_HEADS=0 \
    nohup bash "$HERE/phase1.sh" \
      >>"$CF401_RESULTS/phase1_k${k}.out" 2>&1 &
  pids+=($!); names+=("k=$k gpu=$gpu")
  # Both arms open the same streaming dataset. Starting them together puts
  # two cold HF readers on one connection.
  sleep "$STAGGER"
done

failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  log "arm ${names[$i]} rc=$rc"
  [ $rc -eq 0 ] || failed=$(( failed + 1 ))
done
log "BOX DONE — $failed arm(s) failed"
[ "$failed" -eq 0 ] || exit 1
