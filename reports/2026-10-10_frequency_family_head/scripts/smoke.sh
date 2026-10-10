#!/bin/bash
# freq_family — the test of one wave on elisa, before the waves of 30,000
# steps. Every file goes to <base>/smoke, which no wave reads.
#
# 1. One wave of 500 steps on one data stream, on GPU 1, with the BLK 200k
#    backbone: the control, the 4 family arms, and the two families of one
#    member (shared_strict_m16, heads_strict_m16).
# 2. The parity: a family of one member against the control. The loss rows,
#    then the final weights.
# 3. The count of rows of each member of each family arm, and the step rate
#    and the GPU memory of the wave.
# 4. The B4 score of 4 heads on 6 configs: one config of each member, and
#    one yearly config. The eval log names the member of each config, and a
#    family of one member gets the score of the control.
#
# Usage, on elisa, from the code folder (deploy.sh):  bash smoke.sh [steps]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${1:-500}"
BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
CODE="${FF_CODE:-$BASE/code}"
SMOKE="$BASE/smoke"
BB="${FF_SMOKE_BB:-$HOME/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k/leg_665k/cf419_cos200k_r2_200k.pth}"
ONE="shared_strict_m16 heads_strict_m16"
FILTER='^(bizitobs_service/short|ett1/15T/short|ett1/H/short|us_births/D/short|m4_weekly/short|m4_yearly/short)$'

wave(){ FF_BASE="$SMOKE" FF_CODE="$CODE" FF_HEAD_STEPS="$STEPS" \
  FF_HEAD_SAVE_EVERY=1000000 FF_HEAD_LOG_EVERY=100 \
  bash "$HERE/run_wave.sh" blk 200 "$BB" "$@"; }
head_dir(){ echo "$SMOKE/heads/eval/blk_bb200k_h${STEPS}_$1"; }
head_file(){ echo "$(head_dir "$1")/qhead_blk_bb200k_h${STEPS}_$1_s20260722_$2"; }

echo "== 1. the wave of $STEPS steps"
FF_SCORE=0 wave control shared_strict shared_draw heads_strict heads_draw $ONE || exit 1
log="$(ls -t "$SMOKE"/results/waves/*/train.log | head -1)"

echo "== 2. a family of one member against the control"
for arm in $ONE; do
  PYTHONPATH="$CODE" python3 "$HERE/compare_parity.py" losses "$arm" \
    "$(head_file "ff_$arm" losses.csv)" "$(head_file control losses.csv)"
  PYTHONPATH="$CODE" python3 "$HERE/compare_parity.py" heads "$arm" \
    "$(head_file "ff_$arm" final.pth)" "$(head_file control final.pth)"
done

echo "== 3. the rows of each member, the step rate and the GPU memory ($log)"
grep 'rows of each member' "$log"
grep '^\[shared\]' "$log" | tail -2

echo "== 4. the B4 score of 4 heads on 6 configs"
FF_TRAIN=0 FF_EVAL_FILTER="$FILTER" FF_EVAL_EXPECT=6 FF_EVAL_SLOTS=4 \
  wave control shared_draw $ONE || exit 1
grep -h 'family member' "$(head_dir ff_shared_draw)"/gift/shard_0/shard.log
column -t -s "$(printf '\t')" "$SMOKE/results/scores.tsv"
for arm in $ONE; do
  PYTHONPATH="$CODE" python3 "$HERE/compare_parity.py" tables "$arm" \
    "$(head_dir "ff_$arm")/gift/all_results.csv" "$(head_dir control)/gift/all_results.csv"
done
