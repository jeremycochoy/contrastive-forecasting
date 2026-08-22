#!/bin/bash
# #404 — proof that the L_align weight the arms table names is the weight the
# guard reads back, and that a weight which did not reach the trainer stops
# the leg.
#
# THE DEFECT THIS TEST GUARDS. `w3_s08` differs from `s08` in ONE column of
# `arms.tsv`: the align weight, 3.0 against the cell's 1.0. Every other value
# is equal, down to the backbone seed. So a weight that does not reach the
# trainer trains `s08` a second time under the name `w3_s08`, and the card
# reads a deliberate change to the objective as a repeat.
#
# `run_arm.sh` guards it the way it guards alpha and the seed: off the
# trainer's own command line. That reader is not the one the other three use.
# The cell states `--align-loss-weight 1.0` in its own flag block and an arm
# that moves the weight REPEATS the flag at the end of the line, where
# argparse keeps it. `cf404_arg_of_cmdline` stops at the FIRST hit, so it would
# report 1.0 for every arm and the guard would pass on a leg that trained the
# wrong objective. `cf404_align_of_cmdline` reads the LAST hit.
#
# Usage: bash scripts/test_align_guard.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

pass=0; fail=0
ok(){  printf '  PASS  %s\n' "$1"; pass=$(( pass + 1 )); }
bad(){ printf '  FAIL  %s\n' "$1"; fail=$(( fail + 1 )); }
eq(){  # <label> <got> <want>
  if [ "$2" = "$3" ]; then ok "$1 -> $2"; else bad "$1 -> '$2', wanted '$3'"; fi
}

# The trainer command line of this cell, in the order `run_leg_k.sh` builds
# it: the cell's own weight early, GAP_ARGS last.
LINE_CELL="python3 -u train.py --qk-norm --align-loss-weight 1.0 --moco-rep-keys \
--tau-rep 1.0 --align-target teacher --seed 20260520 --train-rollout-depth 32 \
--cpc-infonce-weight 0.0 --train-rollout-reduce mean"
LINE_W3="$LINE_CELL --align-loss-weight 3.0"

echo "the reader takes the LAST value, which is the one argparse keeps"
eq "the cell alone" "$(printf '%s' "$LINE_CELL" | cf404_align_of_cmdline)" "1.0"
eq "the cell then 3.0" "$(printf '%s' "$LINE_W3" | cf404_align_of_cmdline)" "3.0"
eq "no flag at all" "$(printf '%s' "python3 train.py --tau 0.1" | cf404_align_of_cmdline)" "-"
# The `=` form, which argparse also accepts.
eq "the = form, last" \
   "$(printf '%s' "$LINE_CELL --align-loss-weight=3.0" | cf404_align_of_cmdline)" "3.0"
# A /proc cmdline is NUL separated, not space separated.
eq "a NUL-separated cmdline" \
   "$(printf 'python3\0--align-loss-weight\0001.0\0--align-loss-weight\0003.0' \
      | cf404_align_of_cmdline)" "3.0"

echo
echo "the first-hit reader is the defect this exists for"
first="$(printf '%s' "$LINE_W3" | cf404_arg_of_cmdline --align-loss-weight)"
if [ "$first" = "1.0" ]; then
  ok "cf404_arg_of_cmdline reads 1.0 on the w3 line, so it may not be used here"
else
  bad "cf404_arg_of_cmdline -> '$first' — this test's premise moved"
fi

echo
echo "the table and the command line are compared as NUMBERS"
cf404_num_eq 3.0 3     && ok "3.0 equals 3"          || bad "3.0 does not equal 3"
cf404_num_eq 1.0 1.00  && ok "1.0 equals 1.00"       || bad "1.0 does not equal 1.00"
cf404_num_eq 1.0 3.0   && bad "1.0 equals 3.0"       || ok "1.0 differs from 3.0"
cf404_num_eq - 1.0     && bad "'-' equals 1.0"       || ok "'-' differs from 1.0"

echo
echo "every arm of the table, against the weight its row names"
for arm in $CF404_ARMS; do
  w="$(cf404_align_weight "$arm")"
  case "$arm" in
    w3_s08) want=3.0 ;;
    *)      want=1.0 ;;
  esac
  if cf404_num_eq "$w" "$want"; then ok "$arm align_w=$w"; else bad "$arm align_w=$w, wanted $want"; fi
done

echo
echo "the guard fires on a leg that trained the wrong weight"
# What run_arm.sh compares. `w3_s08` against a command line that carries the
# cell's own weight is the wiring defect, and it must NOT pass.
if cf404_num_eq "$(printf '%s' "$LINE_CELL" | cf404_align_of_cmdline)" \
                "$(cf404_align_weight w3_s08)"; then
  bad "w3_s08 accepts a 1.0 command line"
else
  ok "w3_s08 refuses a 1.0 command line"
fi
if cf404_num_eq "$(printf '%s' "$LINE_W3" | cf404_align_of_cmdline)" \
                "$(cf404_align_weight w3_s08)"; then
  ok "w3_s08 accepts a 3.0 command line"
else
  bad "w3_s08 refuses a 3.0 command line"
fi
if cf404_num_eq "$(printf '%s' "$LINE_CELL" | cf404_align_of_cmdline)" \
                "$(cf404_align_weight s08)"; then
  ok "s08 accepts a 1.0 command line"
else
  bad "s08 refuses a 1.0 command line"
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
