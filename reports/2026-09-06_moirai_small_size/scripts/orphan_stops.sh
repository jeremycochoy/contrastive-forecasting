#!/bin/bash
# #414 — every backbone stop that holds a checkpoint and no score.
#
# WHY THIS EXISTS. On 09-14 `phase1.sh` marked the 400,000-step leg of
# `fix099_dec10k` and of `fix09_dec10k` FAILED rc=2 AFTER each one reached
# 400,000 steps and wrote a valid checkpoint. It then skipped the head. Both
# scores were lost in silence, and a stop with no score is invisible in
# `scores.csv`.
#
# It prints one line for each orphan. It starts nothing: a stop already
# covered by a running head must not get a second one.
#
# Usage:  bash scripts/orphan_stops.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CF412_CKPT_ROOT:-/home/jupyter/checkpoints_backup/cf-412}"
RESULTS="$HERE/../results"
CELL=arm6_v2_combab_alignT
n=0
for leg in "$ROOT"/*/"$CELL"/leg_*k; do
  [ -d "$leg" ] || continue
  arm="${leg#"$ROOT"/}"; arm="${arm%%/*}"
  stop="$(basename "$leg")"; stop="${stop#leg_}"
  ls "$leg"/*_"$stop".pth >/dev/null 2>&1 || continue
  [ -s "$RESULTS/score_${arm}_bb${stop}_h30k_student.txt" ] && continue
  # A head or an evaluation already in flight is not an orphan.
  ps -eo args --no-headers | grep -q "[h]ead_eval.sh $arm ${stop%k}000" && continue
  echo "ORPHAN $arm $stop"
  n=$(( n + 1 ))
done
echo "$n orphan stop(s)"
