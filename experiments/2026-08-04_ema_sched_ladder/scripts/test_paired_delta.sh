#!/bin/bash
# #393 — guard for scripts/paired_delta.py.
#
# Usage:  bash scripts/test_paired_delta.sh
#
# The file it guards is the fix for the thing that blocked the study: a
# significance computed by dividing a paired delta by the spread of ONE of
# its two ends. Four ways that can come back, all of them silent:
#
#   * the denominator quietly reverting to the bb100k spread, which makes a
#     delta with a noisy bb40k end look 10x more certain than it is;
#   * a GPU correction applied to a pair that does not need one, or skipped
#     on a pair that does;
#   * a cell with one seed reporting a significance at all;
#   * a sign that flips across seeds being reported as a mean with a t,
#     when what the extend rule reads is the sign.
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

put(){  # <cell> <stop_k> <enc> <seed> <value> [tag]
  local sfx=""; [ "$4" != 20260722 ] && sfx="_s$4"
  [ -n "${6:-}" ] && sfx="${sfx}_$6"
  mkdir -p "$RUNS/$1/eval"
  printf '%s\n' "$5" > "$RUNS/$1/eval/score_bb$2k_$3$sfx.txt"
}

col(){  # <file> <cell> <head-or-empty> <column>
  awk -F, -v c="$2" -v h="$3" -v want="$4" '
    NR==1{for(i=1;i<=NF;i++) k[$i]=i; next}
    $1==c && (h=="" || $k["head"]==h){print $k[want]}' "$OUTD/$1"
}

# paired_delta.py knows the real cells by name, and the GPU map is keyed on
# those names, so the fixtures have to use them. A is 4090-native (no
# correction is owed), C is one of the three 5090 cells (one is).
A=arm6_v2_combab_alignS      # clean: both ends replicated, delta stable
B=arm6_v2_combab_alignT      # tight bb100k, wide bb40k — the review's case
C=arm5_combab_alignS         # 5090 cell: the GPU control has to be applied
D=arm5_combab_alignT         # bb40k end still training

# --- A: both ends at three seeds, the delta holds its sign and its size ---
put $A 40  student 20260722 1.1603; put $A 40  teacher 20260722 1.1544
put $A 40  student 20260723 1.1600; put $A 40  teacher 20260723 1.1540
put $A 40  student 20260724 1.1610; put $A 40  teacher 20260724 1.1550
put $A 100 student 20260722 1.1945; put $A 100 teacher 20260722 1.1837
put $A 100 student 20260723 1.1898; put $A 100 teacher 20260723 1.1829
put $A 100 student 20260724 1.1925; put $A 100 teacher 20260724 1.1909
# A stray control on a cell whose two ends already share a card. It must be
# ignored: correcting a matched pair would invent a shift that is not there.
put $A 40  student 20260722 1.9999 hw4090

# --- B: the exact failure the review blocked on -------------------------
# bb100k is tight to 0.0002, so a one-ended sigma reads ~100. bb40k moves by
# 0.08 across seeds, so the paired delta changes sign. Same numbers, opposite
# verdicts, which is the whole point.
put $B 40  student 20260722 1.1895; put $B 40  teacher 20260722 1.1793
put $B 40  student 20260723 1.1500; put $B 40  teacher 20260723 1.1790
put $B 40  student 20260724 1.2300; put $B 40  teacher 20260724 1.1795
put $B 100 student 20260722 1.1921; put $B 100 teacher 20260722 1.1963
put $B 100 student 20260723 1.1920; put $B 100 teacher 20260723 1.1962
put $B 100 student 20260724 1.1922; put $B 100 teacher 20260724 1.1964

# --- C: the GPU control ------------------------------------------------
# bb40k seed 22 on the 5090 is 1.2596; the same seed on a 4090 is 1.2696, so
# the measured term is +0.0100 and every 4090 bb40k value must come down by
# it before being paired with a 5090 bb100k value.
put $C 40  student 20260722 1.2596; put $C 40  teacher 20260722 1.2347
put $C 40  student 20260722 1.2696 hw4090
put $C 40  teacher 20260722 1.2397 hw4090
put $C 40  student 20260723 1.2700; put $C 40  teacher 20260723 1.2400
put $C 40  student 20260724 1.2680; put $C 40  teacher 20260724 1.2410
put $C 100 student 20260722 1.2102; put $C 100 teacher 20260722 1.2407
put $C 100 student 20260723 1.2007; put $C 100 teacher 20260723 1.2555
put $C 100 student 20260724 1.2015; put $C 100 teacher 20260724 1.2349

# --- D: bb100k replicated, bb40k not. One pair only. --------------------
put $D 40  student 20260722 1.3334; put $D 40  teacher 20260722 1.3190
put $D 100 student 20260722 1.2797; put $D 100 teacher 20260722 1.2772
put $D 100 student 20260723 1.2956; put $D 100 teacher 20260723 1.2905
put $D 100 student 20260724 1.2908; put $D 100 teacher 20260724 1.3010

echo "paired_delta.py"
python3 "$HERE/paired_delta.py" --runs "$RUNS" --results "$OUTD" >/dev/null 2>&1
check "exit 0" "0" "$?"

# --- 1. A: three paired deltas, same sign, and a real t -----------------
check "A student n_seeds"      "3"          "$(col paired_delta.csv $A student n_seeds)"
check "A student delta s22"    "+0.034200"  "$(col paired_delta.csv $A student delta_20260722)"
check "A student delta s23"    "+0.029800"  "$(col paired_delta.csv $A student delta_20260723)"
check "A student delta s24"    "+0.031500"  "$(col paired_delta.csv $A student delta_20260724)"
check "A student mean delta"   "+0.031833"  "$(col paired_delta.csv $A student delta_mean)"
check "A student sign stable"  "yes"        "$(col paired_delta.csv $A student sign_stable)"
check "A student n_down"       "0/3"        "$(col paired_delta.csv $A student n_down)"
check "A student df"           "2"          "$(col paired_delta.csv $A student df)"
check "A is not GPU-corrected" "no"         "$(col paired_delta.csv $A student hw_corrected)"
check "A has no control row"   ""           "$(col hw_control.csv $A student hw_offset)"

# The SE is the spread of the three DELTAS over sqrt(3), recomputed here from
# the delta columns alone. A regression to the bb100k-only denominator fails
# this line and nothing else in the file would.
want_se="$(python3 - "$OUTD/paired_delta.csv" $A student <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
r = next(x for x in rows if x["cell"] == sys.argv[2] and x["head"] == sys.argv[3])
d = [float(r[f"delta_{s}"]) for s in ("20260722", "20260723", "20260724")]
print(f"{statistics.stdev(d) / len(d) ** 0.5:.6f}")
PY
)"
check "A student se is the paired one" "$want_se" \
  "$(col paired_delta.csv $A student se_paired)"

# --- 2. B: the one-ended denominator would have called this certain ------
# bb100k sd is 0.0001, so |delta_mean| / sd_bb100k is in the tens. The paired
# SE is ~0.023 and the sign is not even constant.
check "B student sign flips"   "no"   "$(col paired_delta.csv $B student sign_stable)"
check "B student n_down"       "1/3"  "$(col paired_delta.csv $B student n_down)"
check "B student is not significant" "no" \
  "$(col paired_delta.csv $B student significant)"
if grep -q "sign flips" <(col paired_delta.csv $B student verdict); then
  ok "B verdict says the sign flips"
else bad "B verdict: $(col paired_delta.csv $B student verdict)"; fi
# The guard that matters: paired SE must be far larger than the bb100k-only
# one on this fixture, not equal to it.
python3 - "$OUTD/paired_delta.csv" $B student <<'PY'
import csv, sys
r = next(x for x in csv.DictReader(open(sys.argv[1]))
         if x["cell"] == sys.argv[2] and x["head"] == sys.argv[3])
se_paired, sd100 = float(r["se_paired"]), float(r["bb100k_sd"])
sys.exit(0 if se_paired > 20 * sd100 else 1)
PY
if [ $? -eq 0 ]; then ok "B paired SE dwarfs the bb100k-only spread"
else bad "B paired SE did not carry the bb40k end"; fi
# And the branch the rule would read flips with it.
check "B branch at s22" "none_down" "$(col paired_branches.csv $B "" branch_20260722)"
check "B branch at s23" "none_down" "$(col paired_branches.csv $B "" branch_20260723)"
check "B branch at s24" "one_down"  "$(col paired_branches.csv $B "" branch_20260724)"
check "B does not survive" "no" "$(col paired_branches.csv $B "" survives_paired)"

# --- 3. C: the GPU control is measured and applied ----------------------
check "C student hw offset"  "+0.010000" "$(col hw_control.csv $C student hw_offset)"
check "C teacher hw offset"  "+0.005000" "$(col hw_control.csv $C teacher hw_offset)"
check "C student is corrected" "yes" "$(col paired_delta.csv $C student hw_corrected)"
check "C s22 bb40k is untouched" "1.259600" \
  "$(col paired_delta.csv $C student bb40k_20260722)"
check "C s23 bb40k is shifted onto the 5090" "1.260000" \
  "$(col paired_delta.csv $C student bb40k_20260723)"
check "C s23 delta uses the shifted value" "-0.059300" \
  "$(col paired_delta.csv $C student delta_20260723)"
check "C student holds its sign" "3/3" "$(col paired_delta.csv $C student n_down)"

# --- 4. D: one pair is not a significance -------------------------------
check "D student n_seeds"    "1" "$(col paired_delta.csv $D student n_seeds)"
check "D student se is blank" "" "$(col paired_delta.csv $D student se_paired)"
check "D student t is blank"  "" "$(col paired_delta.csv $D student t_paired)"
check "D student significance is blank" "" \
  "$(col paired_delta.csv $D student significant)"
if grep -q "no claim yet" <(col paired_delta.csv $D student verdict); then
  ok "D verdict withholds the claim"
else bad "D verdict: $(col paired_delta.csv $D student verdict)"; fi
check "D branch row withholds too" "" \
  "$(col paired_branches.csv $D "" survives_paired)"

# --- 5. file hygiene ----------------------------------------------------
check "no CR in any output" "0" \
  "$(cat "$OUTD"/paired_*.csv "$OUTD"/hw_control.csv | tr -cd '\r' | wc -c)"
check "hw_control lists only the mismatched cells" "6" \
  "$(( $(wc -l <"$OUTD/hw_control.csv") - 1 ))"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
