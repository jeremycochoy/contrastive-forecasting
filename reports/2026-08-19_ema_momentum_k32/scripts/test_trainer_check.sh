#!/bin/bash
# #404 — proof that the box's "a trainer already runs" check reads FALSE on a
# box with no trainer.
#
# THE DEFECT THIS TEST GUARDS. `round2_box.sh` asks the box whether a trainer
# is already up before it starts one. It asks over SSH:
#
#     ssh box "pgrep -f 'run_leg_k.sh arm6_v2_combab_alignT' >/dev/null"
#
# sshd runs that string through a shell, so the box holds a process whose
# command line IS the pattern. `pgrep -f` reads full command lines. It drops
# its own pid and nothing else, so it matches that shell. The check is true on
# a bare box, the driver starts no trainer, and the box bills at 0% GPU.
#
# On 2026-08-19 this cost three boxes, 30 minutes each, and returned no data.
#
# THE FIX. `cf404_pgrep_pattern` puts a bracket class on the first character.
# `[r]un_leg_k` matches the text `run_leg_k`. It does not match the text
# `[r]un_leg_k` that the shell carries.
#
# WHY A NONCE. A bare pattern also matches processes this study does not own.
# On elisa, `train_forecasting_head` matches two runs of the #401 session AND
# the agent shell, whose prompt merely names the file. So the assertions below
# carry a per-run nonce: nothing outside this test can match them, and a
# failure is the defect and never a neighbour. The two REAL driver patterns are
# then checked as well, against the machine as it is.
#
# WHAT THIS TEST RUNS. `bash -c "<command>"` is what sshd does with a remote
# command, so the local shell reproduces the box exactly.
#
# Usage: bash scripts/test_trainer_check.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

TMP="$(mktemp -d)"
NONCE="cf404test$$"
trap 'rm -rf "$TMP"' EXIT

pass=0; fail=0
ok(){   printf '  PASS  %s\n' "$1"; pass=$(( pass + 1 )); }
bad(){  printf '  FAIL  %s\n' "$1"; fail=$(( fail + 1 )); }

# The box's own question, asked through the shell sshd would use.
asks(){  # <pgrep pattern>
  bash -c "pgrep -f '$1' >/dev/null"
}
check(){  # <label> <expected: yes|no> <pgrep pattern>
  local got
  if asks "$3"; then got=yes; else got=no; fi
  if [ "$got" = "$2" ]; then ok "$1 -> $got"; else bad "$1 -> $got, wanted $2"; fi
}

# A process whose command line looks like the trainer's. The name sits in the
# ARGUMENTS, which is where the real trainer carries it and where `pgrep -f`
# looks.
fake(){  # <script name> <arg...> -> prints the pid
  local name="$1"; shift
  printf '#!/bin/bash\nsleep 600\n' >"$TMP/$name"
  chmod +x "$TMP/$name"
  bash "$TMP/$name" "$@" >/dev/null 2>&1 &
  echo $!
}

# The two shapes the driver sends, with the nonce that makes them private.
BB_RAW="run_leg_k.sh $CF404_CELL"
HEAD_RAW="train_forecasting_head"
BB_N="run_leg_k.sh $CF404_CELL $NONCE"
HEAD_N="train_forecasting_head $NONCE"
BB_PAT="$(cf404_pgrep_pattern "$BB_N")"
HEAD_PAT="$(cf404_pgrep_pattern "$HEAD_N")"

echo "#404 trainer-check test   nonce=$NONCE"
echo "  backbone pattern: $BB_PAT"
echo "  head pattern:     $HEAD_PAT"

echo
echo "1. no trainer — the check must be false"
check "backbone, fixed pattern" no "$BB_PAT"
check "head, fixed pattern"     no "$HEAD_PAT"

echo
echo "2. the defect, kept visible — a bare pattern is true with no trainer"
# `bash -c` puts the bare pattern in the shell's own command line, and pgrep -f
# matches it. This assertion failing would mean the shell stopped carrying its
# command line, not that a bare pattern became safe.
check "backbone, bare pattern" yes "$BB_N"
check "head, bare pattern"     yes "$HEAD_N"

echo
echo "3. a trainer runs — the check must be true"
bb_pid="$(fake run_leg_k.sh "$CF404_CELL" "$NONCE")"
head_pid="$(fake train_forecasting_head "$NONCE")"
sleep 1
check "backbone, fixed pattern" yes "$BB_PAT"
check "head, fixed pattern"     yes "$HEAD_PAT"
# By pid, never by pattern: this test starts the two processes and knows them.
kill "$bb_pid" "$head_pid" 2>/dev/null
wait "$bb_pid" 2>/dev/null; wait "$head_pid" 2>/dev/null

echo
echo "4. the trainers are gone — the check must be false again"
sleep 1
check "backbone, fixed pattern" no "$BB_PAT"
check "head, fixed pattern"     no "$HEAD_PAT"

echo
echo "5. the real driver patterns, against this machine as it is"
# No nonce here. A `yes` is only wrong when nothing is really training, so the
# result is printed with the count of processes that honestly match.
for raw in "$BB_RAW" "$HEAD_RAW"; do
  pat="$(cf404_pgrep_pattern "$raw")"
  n="$(pgrep -f "$pat" 2>/dev/null | wc -l | tr -d ' ')"
  if asks "$pat"; then verdict="true"; else verdict="false"; fi
  printf '  INFO  %-42s -> %-5s  %s real process(es)\n' "$pat" "$verdict" "$n"
  bare="$(pgrep -f "$raw" 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$bare" -gt "$n" ]; then
    ok "the bare pattern over-matches by $(( bare - n )) here, the fixed one does not"
  fi
done

echo
echo "6. the driver sends the fixed pattern, not a bare one"
for line in 137 181; do
  src="$(sed -n "${line}p" "$HERE/round2_box.sh")"
  case "$src" in
    *cf404_pgrep_pattern*) ok "round2_box.sh:$line uses cf404_pgrep_pattern" ;;
    *pgrep*) bad "round2_box.sh:$line sends a bare pattern: $src" ;;
    *) bad "round2_box.sh:$line is not the check any more: $src" ;;
  esac
done

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
