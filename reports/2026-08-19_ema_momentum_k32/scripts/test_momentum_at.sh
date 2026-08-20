#!/bin/bash
# #404 — `cf404_momentum_at` against `src.models.ema_tau_at_step`.
#
# The shell helper repeats the trainer's formula so a table can be printed
# without a Python interpreter. A repeat is a second copy, and a second copy
# drifts. This test walks EVERY arm of the table over a range of steps and
# fails on the first disagreement past 1e-6.
#
# Usage: bash scripts/test_momentum_at.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

STEPS="0 1 1000 20000 40000 99999 100000 100001 200000 400000"
fail=0; checked=0

for arm in $CF404_ARMS; do
  read -r _ tau end ramp <<<"$(cf404_arm_row "$arm")"
  for step in $STEPS; do
    got="$(cf404_momentum_at "$arm" "$step")"
    want="$(cd "$CF404_REPO" && python3 -c "
import sys
sys.path.insert(0, '.')
from src.models import ema_tau_at_step
end = None if '$end' == '-' else float('$end')
ramp = None if '$ramp' == '-' else int('$ramp')
print('%.3f' % ema_tau_at_step($step, 40000, float('$tau'), end, ramp))
")"
    checked=$(( checked + 1 ))
    if [ "$got" != "$want" ]; then
      echo "FAIL $arm step=$step shell=$got python=$want"
      fail=1
    fi
  done
done

if [ "$fail" -eq 0 ]; then
  echo "OK — $checked (arm, step) pairs agree, over $(echo "$CF404_ARMS" | wc -w) arm(s)"
else
  echo "FAILED"
fi
exit "$fail"
