#!/bin/bash
# #393 — guard for scripts/seed_spread.py.
#
# Usage:  bash scripts/test_seed_spread.sh
#
# The file it guards decides what the study is allowed to claim: whether each
# cell's extend-rule branch survives the head-seed spread, or was decided by
# noise. Three ways that can go wrong, all of them silent:
#
#   * calling a branch SURVIVED when a seed is simply still training;
#   * calling a change RESOLVED off one seed, whose range is zero;
#   * saying a branch flipped without saying which head's sign moved, which
#     is the part a reader needs and the part the numbers actually pin down.
#
# Fixtures are score files in a throwaway RUNS tree, the same layout the real
# run writes, so the reader under test is the real one.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
RUNS="$TMP/runs"; OUTD="$TMP/results"; mkdir -p "$OUTD"
pass=0; fail=0

ok(){   pass=$(( pass + 1 )); echo "  ok   $*"; }
bad(){  fail=$(( fail + 1 )); echo "  FAIL $*"; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1: want '$2', got '$3'"; fi; }

put(){  # <cell> <stop_k> <enc> <seed> <value>
  local sfx=""; [ "$4" != 20260722 ] && sfx="_s$4"
  mkdir -p "$RUNS/$1/eval"
  printf '%s\n' "$5" > "$RUNS/$1/eval/score_bb$2k_$3$sfx.txt"
}

col(){  # <file> <cell> <head-or-empty> <column>
  awk -F, -v c="$2" -v h="$3" -v want="$4" '
    NR==1{for(i=1;i<=NF;i++) k[$i]=i; next}
    $1==c && (h=="" || $k["head"]==h){print $k[want]}' "$OUTD/$1"
}

# --- the fixtures --------------------------------------------------------
# A. every seed says the same thing: both heads up, none_down, and the moves
#    are far bigger than the spread.
put A 40 student 20260722 1.0000; put A 40 teacher 20260722 1.0000
for s in 20260722 20260723 20260724; do
  put A 100 student "$s" 1.3000; put A 100 teacher "$s" 1.3000
done
put A 100 student 20260723 1.3010; put A 100 teacher 20260724 1.2990

# B. the branch the head seed flips. bb40k 1.20; seed 22 lands just below on
#    both heads (both_down, extend), the other two land just above.
put B 40 student 20260722 1.2000; put B 40 teacher 20260722 1.2000
put B 100 student 20260722 1.1990; put B 100 teacher 20260722 1.1990
put B 100 student 20260723 1.2050; put B 100 teacher 20260723 1.2040
put B 100 student 20260724 1.2030; put B 100 teacher 20260724 1.2060

# C. only ONE head's sign moves. The student is below bb40k on all three
#    seeds; the teacher is below on two and above on the third. The branch
#    goes both_down -> one_down, and the report has to be able to say it was
#    the teacher that moved.
put C 40 student 20260722 1.3000; put C 40 teacher 20260722 1.3000
put C 100 student 20260722 1.2900; put C 100 teacher 20260722 1.2900
put C 100 student 20260723 1.2800; put C 100 teacher 20260723 1.2950
put C 100 student 20260724 1.2850; put C 100 teacher 20260724 1.3100

# D. still training: only the protocol seed is scored.
put D 40 student 20260722 1.4000; put D 40 teacher 20260722 1.4000
put D 100 student 20260722 1.3000; put D 100 teacher 20260722 1.3000

# seed_spread.py knows the six real cells by name, so the fixtures borrow
# them. The mapping is only for this test.
declare -A NAME=( [A]=arm6_v2_combab_alignS [B]=arm6_v2_combab_alignT
                  [C]=arm5_combab_alignS    [D]=arm5_combab_alignT )
for f in A B C D; do mv "$RUNS/$f" "$RUNS/${NAME[$f]}"; done

echo "seed_spread.py"
python3 "$HERE/seed_spread.py" --runs "$RUNS" --results "$OUTD" >/dev/null 2>&1
check "exit 0" "0" "$?"

A=${NAME[A]}; B=${NAME[B]}; C=${NAME[C]}; D=${NAME[D]}

# 1. A clean cell: both heads are ABOVE bb40k on every seed, so none_down
#    holds and neither down-list has an entry.
check "A recorded branch"        "none_down" "$(col seed_branches.csv "$A" "" recorded_branch)"
check "A survives the seeds"     "yes"       "$(col seed_branches.csv "$A" "" survives_matched)"
check "A: no seed puts the student below bb40k" "" \
  "$(col seed_branches.csv "$A" "" student_down_seeds)"
check "A: no seed puts the teacher below bb40k" "" \
  "$(col seed_branches.csv "$A" "" teacher_down_seeds)"
check "A change is resolved"     "yes"       "$(col seed_spread.csv "$A" student resolved)"

# 2. The flip. This is the whole point of the replicate.
check "B recorded branch"        "both_down" "$(col seed_branches.csv "$B" "" recorded_branch)"
check "B flips on seed 20260723" "none_down" "$(col seed_branches.csv "$B" "" branch_20260723)"
check "B does not survive"       "no"        "$(col seed_branches.csv "$B" "" survives_matched)"
check "B change is not resolved" "no"        "$(col seed_spread.csv "$B" student resolved)"
if grep -q "FLIPS" <(col seed_branches.csv "$B" "" verdict); then
  ok "B verdict says it flips"
else bad "B verdict: $(col seed_branches.csv "$B" "" verdict)"; fi

# 3. One head's sign moves. The flip has to be attributed to that head.
check "C does not survive"        "no"        "$(col seed_branches.csv "$C" "" survives_matched)"
check "C ends at one_down"        "one_down"  "$(col seed_branches.csv "$C" "" branch_20260724)"
check "C student holds its sign"  "20260722 20260723 20260724" \
  "$(col seed_branches.csv "$C" "" student_down_seeds)"
check "C teacher loses one seed"  "20260722 20260723" \
  "$(col seed_branches.csv "$C" "" teacher_down_seeds)"
if grep -q "sign on the teacher" <(col seed_branches.csv "$C" "" verdict); then
  ok "C verdict names the teacher as what moved"
else bad "C verdict: $(col seed_branches.csv "$C" "" verdict)"; fi

# 4. Incomplete is not the same as flipped, and one seed has no spread.
check "D does not claim to survive" "" "$(col seed_branches.csv "$D" "" survives_matched)"
check "D resolved is blank on one seed" "" "$(col seed_spread.csv "$D" student resolved)"
if grep -q "not enough seeds" <(col seed_branches.csv "$D" "" verdict); then
  ok "D verdict says it is unfinished"
else bad "D verdict: $(col seed_branches.csv "$D" "" verdict)"; fi

# 5. The spread itself. A's student seeds are 1.3000 1.3010 1.3000.
check "A student mean"  "1.300333" "$(col seed_spread.csv "$A" student mean)"
check "A student range" "0.001000" "$(col seed_spread.csv "$A" student range)"
check "A student n"     "3"        "$(col seed_spread.csv "$A" student n_seeds)"

# 6. The auditable row table carries the replicates and NOT the protocol
#    seed, which ladder_all.csv already holds and audit_scores.py already
#    checks there. Four cells: A B C have 4 replicate rows each, D none.
check "replicate rows only" "12" "$(( $(wc -l <"$OUTD/seed_spread_rows.csv") - 1 ))"
check "no protocol-seed row" "0" \
  "$(awk -F, 'NR>1 && $8==20260722' "$OUTD/seed_spread_rows.csv" | wc -l)"
check "head_seed column is present" \
  "cell,arm,align,stop,head,head_steps,ema_tau,head_seed,gm_rel_mase" \
  "$(head -1 "$OUTD/seed_spread_rows.csv")"
check "no CR in any output" "0" \
  "$(cat "$OUTD"/seed_*.csv | tr -cd '\r' | wc -c)"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
