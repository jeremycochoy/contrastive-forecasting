#!/bin/bash
# #412 — is a trainer already running this arm?
#
# WHY IT EXISTS. Nothing below this card stops two trainers on one arm.
# `run_leg_k.sh` skips only when the TARGET checkpoint is already on disk,
# which protects nothing while a leg runs. Its cell claim reads
# `results/cell_claims.txt`, which no study of this line creates, so the block
# never executes. It also keys on the CELL, and all ten arms of this card
# share `arm6_v2_combab_alignT`, so it could not separate them in any case.
#
# Two trainers on one arm write one `leg_40k` directory and one losses CSV.
# The result is an arm whose curve is two runs interleaved, and no file says
# so.
#
# WHAT IT READS. The run name, which IS unique per arm: `run_arm.sh` passes
# `RUN_SUFFIX=_cf412_<arm>`, and the trainer carries `--run-name
# cf393_<cell>_cf373k<k>_cf412_<arm>` on its command line. The match is
# anchored at the end, so `_cf412_k3_r100_09` does not match
# `_cf412_k3_r100_09_dec`.
#
# Exit 0 when the arm is busy, 1 when it is free.
#
# Usage:  bash scripts/arm_busy.sh k3_r100_09_lr17 || start the leg
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

ARM="${1:?usage: arm_busy.sh <arm>}"
cf412_require_arm "$ARM" || exit 2
K="$(cf412_depth "$ARM")" || exit 2
NAME="$(printf 'cf393_%s_cf373k%s_cf412_%s' "$CF412_CELL" "$K" "$ARM")"

# `--run-name <name>` followed by a space or the end of the command line.
hit="$(pgrep -a -f -- "--run-name ${NAME}( |\$)" 2>/dev/null \
       | grep -v arm_busy | head -3)"
if [ -n "$hit" ]; then
  echo "BUSY: a trainer already runs $ARM"
  echo "$hit" | cut -c1-120
  exit 0
fi
echo "FREE: no trainer runs $ARM"
exit 1
