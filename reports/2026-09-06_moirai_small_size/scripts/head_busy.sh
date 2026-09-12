#!/bin/bash
# #412 — is a head or an evaluation already running on this checkpoint?
#
# WHY IT EXISTS. `missing_heads.sh` asks `pgrep -f "head_eval.sh <arm>
# <stop>"`, and `pgrep -f` matches the WHOLE command line of every process on
# the box. Three agent sessions share this machine, and each one runs shell
# commands that name these scripts and these checkpoints while it watches
# them. So the guard matched a peer's `bash -c` watcher and reported a head
# that did not exist.
#
# It cost a head at 06:02:48 on 2026-09-07: `k3_r100_09_dec` reached 40,000
# steps, and both of this session's lanes read "a head already runs it" from
# two watcher shells of session 9532bc. No head was running, and the claim
# lane dropped the arm.
#
# THE FIX IS A WHITELIST, not a longer blacklist. A head is one of exactly
# three processes, and each has a shape that a watcher shell does not:
#
#   1. the driver   bash <path>/head_eval.sh <arm> <stop>
#   2. the trainer  python .. train_forecasting_head.py .. --backbone-path <bb>
#   3. the eval     python .. eval_gift_eval_official.py .. <bb>
#
# A `bash -c` wrapper fails all three, because its second field is `-c`.
#
# Exit 0 when the checkpoint is busy, 1 when it is free.
#
# Usage:  bash scripts/head_busy.sh k3_r100_09_dec 40000
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARM="${1:?usage: head_busy.sh <arm> <stop>}"
STOP="${2:?usage: head_busy.sh <arm> <stop>}"
cf412_require_arm "$ARM" || exit 2
cf412_require_stop "$STOP" || exit 2
BB="$(cf412_bb_ckpt "$ARM" "$STOP")"
BASE="${BB##*/}"

hits="$(ps -eo pid,args --no-headers 2>/dev/null | awk -v arm="$ARM" \
  -v stop="$STOP" -v base="$BASE" '
  # `ps -eo pid,args` gives pid in $1, the executable in $2 and its first
  # argument in $3. A watcher shell is `/bin/bash -c ...`, so $3 is "-c".
  # Every rule below also pins $2, because a shell that merely NAMES one of
  # these scripts in its own command line must not count as one.
  { line = $0; exe = $2; arg1 = $3
    if (arg1 == "-c") next
    if (exe ~ /bash$/ && arg1 ~ /head_eval\.sh$/ \
        && line ~ ("head_eval\\.sh " arm " " stop "( |$)")) {
      print "driver  " line; next }
    if (exe ~ /python/ && line ~ /train_forecasting_head\.py/ \
        && base != "" && index(line, base)) {
      print "head    " line; next }
    if (exe ~ /python/ && line ~ /eval_gift_eval_official\.py/ \
        && base != "" && index(line, base)) {
      print "eval    " line; next }
  }' | cut -c1-140)"

if [ -n "$hits" ]; then
  echo "BUSY: $ARM at $STOP"; echo "$hits"; exit 0
fi
echo "FREE: $ARM at $STOP has no head and no evaluation"
exit 1
