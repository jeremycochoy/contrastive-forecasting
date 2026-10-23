#!/bin/bash
# freq_family — the B4 score of the standard head of BLK 200k on a GPU of
# elisa: its time, its GPU memory, and its MASE of each config against the
# table of the CPU score of the same head (the box, 10-05: 1.1262).
#
# The eval runs on the GPU for the waves when the two tables agree. The
# script trains no head: it reads the head and the table that elisa holds.
# Every file goes to <base>/timing.
#
# The check runs one time in that folder. eval_local.sh skips a score that
# exists and goes on from the shard tables, so a second start would time a
# skip or a part of the eval. The script refuses it.
#
# Usage, on elisa, from the code folder (deploy.sh):  bash b4_gpu_check.sh
#   FF_GPU=<N>          the GPU (default 1)
#   FF_EVAL_SHARDS=<n>  the shards of the score (default 4)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
. "$HERE/code_folder.sh"
GPU="${FF_GPU:-1}"
T="$BASE/timing"
BLK="$HOME/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k"
BB="$BLK/leg_665k/cf419_cos200k_r2_200k.pth"
OLD="$BLK/eval/cf419cos_bb200k_h30k_student"
HEAD="$OLD/qhead_cf419cos_bb200k_h30k_student_s20260722_final.pth"
OUT="$T/blk200_b4_gpu"

for f in "$T/times.txt" "$T/score_blk200_b4_gpu.txt" "$OUT/gift"; do
  [ ! -e "$f" ] || {
    echo "ABORT: $f exists: this check ran in $T, and a second start times no full score. Its times are in $T/times.txt. Give another FF_BASE to time a score again." >&2
    exit 3; }
done
mkdir -p "$OUT" || exit 2
# Every 10 s: the memory in use (MiB) and the use (%) of the GPU.
( while :; do
    echo "$(date +%T) $(nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ',')"
    sleep 10
  done ) >"$T/gpu$GPU.log" &
sampler=$!
date '+%m-%d %H:%M:%S start' >"$T/times.txt"
WT="$CODE" EVAL_DEVICE=cuda BB_GPU="$GPU" EVAL_SHARDS="${FF_EVAL_SHARDS:-4}" \
  CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" \
  CF393_EVAL_SLOTDIR="$T/evalslots" \
  GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}" \
  bash "$CODE/reports/2026-08-08_rollout_depth/scripts/eval_local.sh" \
  blk200_b4_gpu 200 student "$BB" "$HEAD" "$OUT" "$T/score_blk200_b4_gpu.txt"
echo "rc=$?" >>"$T/times.txt"
date '+%m-%d %H:%M:%S end' >>"$T/times.txt"
kill "$sampler"

cat "$T/times.txt"
echo "GM-Relative MASE on GPU $GPU: $(cat "$T/score_blk200_b4_gpu.txt")"
awk '{ if ($2 > m) m = $2; u += $3; n++ }
  END { printf "peak GPU memory %d MiB, mean GPU use %.0f%%\n", m, u / n }' "$T/gpu$GPU.log"
python3 "$HERE/compare_parity.py" tables "GPU of elisa against the CPU of the box" \
  "$OUT/gift/all_results.csv" "$OLD/gift/all_results.csv"
