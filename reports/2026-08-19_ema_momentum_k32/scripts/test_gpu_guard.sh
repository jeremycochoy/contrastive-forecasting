#!/bin/bash
# #404 — proof that a launcher refuses a card index the box does not carry.
#
# THE DEFECT THIS TEST GUARDS. Round 3's plan print read
#
#     arm a085 gpu=0
#     arm a095 gpu=1
#     arm s08b gpu=0
#
# while the box held ONE card, at index 0. The print came from a dry run that
# passed no GPUS, so `launch_box.sh` took its own default `0 1`. A lane on card
# 1 of a one-card box does not fail at the launch. It fails inside
# `.to(device)`, hours after the operator has left, and the box bills until a
# person looks.
#
# THE FIX. `cf404_require_gpus` reads the card count off the driver and refuses
# every index at or above it. Both launchers call it BEFORE the plan print, so
# a plan that names a card that is not there fails at the print.
#
# WHY AN OVERRIDE. `CF404_GPU_COUNT` stands in for the driver, so the
# assertions below hold on a box with one card, on elisa with two, and on a
# laptop with none. Section 4 then reads the real count of this machine.
#
# Usage: bash scripts/test_gpu_guard.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

pass=0; fail=0
ok(){  printf '  PASS  %s\n' "$1"; pass=$(( pass + 1 )); }
bad(){ printf '  FAIL  %s\n' "$1"; fail=$(( fail + 1 )); }

check(){  # <label> <expected: yes|no> <card count> <indices>
  local got
  if CF404_GPU_COUNT="$3" cf404_require_gpus "$4" >/dev/null 2>&1
  then got=yes; else got=no; fi
  if [ "$got" = "$2" ]; then ok "$1 -> $got"; else bad "$1 -> $got, wanted $2"; fi
}

# A launcher's plan print, which must fail on a bad index and pass on a good
# one. `CF404_DRY_RUN=1` runs every guard and starts nothing.
plan(){  # <script> <card count> <indices>
  CF404_GPU_COUNT="$2" GPUS="$3" ARMS="a095" CF404_DRY_RUN=1 \
    bash "$HERE/$1" >/dev/null 2>&1
}
check_plan(){  # <label> <expected: yes|no> <script> <card count> <indices>
  local got
  if plan "$3" "$4" "$5"; then got=yes; else got=no; fi
  if [ "$got" = "$2" ]; then ok "$1 -> $got"; else bad "$1 -> $got, wanted $2"; fi
}

echo "#404 card-index guard test"

echo
echo "1. one card, index 0 only"
check "gpu 0 on a 1-card box"     yes 1 "0"
check "gpu 1 on a 1-card box"     no  1 "1"
check "gpus '0 1' on a 1-card box" no  1 "0 1"
check "the round-3 default '0 1'"  no  1 "0 1"

echo
echo "2. four cards, indices 0 to 3"
check "gpus '0 1 2 3' on a 4-card box" yes 4 "0 1 2 3"
check "gpu 4 on a 4-card box"          no  4 "4"
check "gpus '0 0 0' on a 1-card box"   yes 1 "0 0 0"

echo
echo "3. what is not an index at all"
check "no card on the box" no 0 "0"
check "a letter"           no 1 "a"
check "a negative index"   no 1 "-1"

echo
echo "4. the launchers, through their own plan print"
check_plan "launch_box.sh, 1 card, gpu 0"       yes launch_box.sh 1 "0"
check_plan "launch_box.sh, 1 card, gpu 1"       no  launch_box.sh 1 "1"
check_plan "launch_box.sh, 1 card, its default" no  launch_box.sh 1 "0 1"
check_plan "heads_box.sh, 1 card, gpu 0"        yes heads_box.sh  1 "0"
check_plan "heads_box.sh, 1 card, gpu 3"        no  heads_box.sh  1 "3"
check_plan "heads_box.sh, 1 card, its default"  no  heads_box.sh  1 "0 1 2 3"

echo
echo "5. the default, when the caller names no card"
# The defect's other half. `launch_box.sh` defaulted to `0 1` and
# `heads_box.sh` to `0 1 2 3`, whatever the box carried. The default is now the
# card list the driver reports, so a plan print with no GPUS names card 0 only
# on a one-card box.
for n in 1 2 4; do
  want="$(CF404_GPU_COUNT=$n bash -c '. '"$HERE"'/study.sh; cf404_default_gpus')"
  case "$n:$want" in
    "1:0"|"2:0 1"|"4:0 1 2 3") ok "$n card(s) -> default '$want'" ;;
    *) bad "$n card(s) -> default '$want'" ;;
  esac
done
lanes="$(CF404_GPU_COUNT=1 ARMS="a095 s08b" CF404_DRY_RUN=1 \
         bash "$HERE/launch_box.sh" 2>/dev/null | awk '/^arm /{print $3}' | sort -u | tr '\n' ' ')"
if [ "$lanes" = "gpu=0 " ]; then
  ok "launch_box.sh on a 1-card box, no GPUS -> every arm on gpu=0"
else
  bad "launch_box.sh on a 1-card box, no GPUS -> '$lanes'"
fi

echo
echo "6. this machine, as the driver reports it"
printf '  INFO  cf404_gpu_count -> %s card(s)\n' "$(cf404_gpu_count)"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
