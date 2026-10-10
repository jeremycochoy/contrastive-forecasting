#!/bin/bash
# freq_family — two probes of the B4 score of the standard head of BLK 200k,
# after b4_gpu_check.sh:
#   cpu    6 configs on one CPU core of elisa. Their MASE against the table
#          of the box shows how near two CPU scores are. Their times against
#          the times of the GPU shards give the time of a full CPU score.
#   tf32   4 configs on the GPU with TF32 off (tf32_off.py). Their MASE
#          against the GPU table shows if TF32 is the cause of the GPU and
#          CPU difference.
# Every file goes to <base>/timing.
#
# Usage, on elisa, from the code folder:  bash b4_probes.sh cpu|tf32
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
. "$HERE/code_folder.sh"
T="$BASE/timing"
BLK="$HOME/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k"
EVAL="$CODE/experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py"
FLAGS=(--backbone-path "$BLK/leg_665k/cf419_cos200k_r2_200k.pth"
       --head-path "$BLK/eval/cf419cos_bb200k_h30k_student/qhead_cf419cos_bb200k_h30k_student_s20260722_final.pth"
       --encoder-source student --strategy B4 --forecast-len 16
       --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 8 --num-layers 3
       --encoder-type gru --rev-norm-kind ewma --rev-norm-span 128
       --head-nhead 8 --head-causal true)
export PYTHONPATH="$CODE" GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case "${1:?usage: b4_probes.sh cpu|tf32}" in
  cpu)
    mkdir -p "$T/cpu_probe"
    CUDA_VISIBLE_DEVICES= python3 -u "$EVAL" "${FLAGS[@]}" --device cpu \
      --output-dir "$T/cpu_probe" \
      --config-filter '^(SZ_TAXI/15T/short|ett1/15T/short|m4_hourly/short|bizitobs_l2c/5T/short|ett1/H/long|M_DENSE/H/long)$' \
      2>&1 | tee "$T/cpu_probe/probe.log" | grep 'MASE=' ;;
  tf32)
    mkdir -p "$T/tf32_probe"
    CUDA_VISIBLE_DEVICES="${FF_GPU:-1}" python3 -u "$HERE/tf32_off.py" "$EVAL" \
      "${FLAGS[@]}" --device cuda --output-dir "$T/tf32_probe/out" \
      --config-filter '^(m4_hourly/short|bizitobs_l2c/5T/short|ett1/15T/short|SZ_TAXI/15T/short)$' \
      2>&1 | tee "$T/tf32_probe/probe.log" | grep 'MASE=' ;;
  *) echo "ABORT: probe '$1'. Use cpu or tf32." >&2; exit 2 ;;
esac
