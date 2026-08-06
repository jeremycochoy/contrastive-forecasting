#!/bin/bash
# #393 — guard for scripts/stop_reason.py.
#
# Usage:  bash scripts/test_stop_reason.sh
#
# The file it guards answers one question the report cannot get wrong:
# which cells the extend rule stopped, and which cells the spend order
# parked. Case 1 is the row that made this necessary — `arm6_v2_ncpc_alignT`
# carrying both `budget_stop` and `none_down` at 100k, in an order sorting
# chose rather than the order the decisions were taken in.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

ok(){   pass=$(( pass + 1 )); echo "  ok   $*"; }
bad(){  fail=$(( fail + 1 )); echo "  FAIL $*"; }
check(){ # <desc> <expected> <actual>
  if [ "$2" = "$3" ]; then ok "$1"; else bad "$1: want '$2', got '$3'"; fi
}

LADDER="$TMP/ladder.csv"
DEC="$TMP/decisions.csv"
OUT="$TMP/stop_reason.csv"

cat >"$LADDER" <<'CSV'
cell,arm,align,stop,head,head_steps,ema_tau,gm_rel_mase
arm6_v2_ncpc_alignT,arm6_v2_ncpc,teacher,40000,student,15000,0.940000,1.295500
arm6_v2_ncpc_alignT,arm6_v2_ncpc,teacher,40000,teacher,15000,0.940000,1.326600
arm6_v2_ncpc_alignT,arm6_v2_ncpc,teacher,100000,student,30000,1.000000,1.390400
arm6_v2_ncpc_alignT,arm6_v2_ncpc,teacher,100000,teacher,30000,1.000000,1.364600
arm5_combab_alignS,arm5_combab,student,40000,student,15000,0.940000,1.259600
arm5_combab_alignS,arm5_combab,student,40000,teacher,15000,0.940000,1.234700
arm5_combab_alignS,arm5_combab,student,100000,student,30000,1.000000,1.210200
arm5_combab_alignS,arm5_combab,student,100000,teacher,30000,1.000000,1.240700
arm5_combab_alignT,arm5_combab,teacher,40000,student,15000,0.940000,1.333400
arm5_combab_alignT,arm5_combab,teacher,40000,teacher,15000,0.940000,1.319000
arm5_combab_alignT,arm5_combab,teacher,100000,student,30000,1.000000,1.279700
arm5_combab_alignT,arm5_combab,teacher,100000,teacher,30000,1.000000,1.277200
arm1_nse,arm1_nse,,40000,student,15000,0.940000,1.434700
arm1_nse,arm1_nse,,40000,teacher,15000,0.940000,1.451200
CSV

# Sorted the way merge_pooled.sh leaves it, NOT the order decided.
cat >"$DEC" <<'CSV'
cell,stop,branch,extend,heads_next
arm1_nse,40000,unconditional,1,student teacher
arm5_combab_alignS,40000,unconditional,1,student teacher
arm5_combab_alignS,100000,budget_stop,0,
arm5_combab_alignS,100000,one_down,1,student
arm5_combab_alignT,40000,unconditional,1,student teacher
arm5_combab_alignT,100000,both_down,1,student teacher
arm5_combab_alignT,100000,session_end,1,student teacher
arm6_v2_ncpc_alignT,40000,unconditional,1,student teacher
arm6_v2_ncpc_alignT,100000,budget_stop,0,
arm6_v2_ncpc_alignT,100000,none_down,0,student teacher
CSV

echo "stop_reason.py"
python3 "$HERE/stop_reason.py" --ladder "$LADDER" --decisions "$DEC" \
        --out "$OUT" --no-probe >/dev/null 2>&1
check "exit 0" "0" "$?"

col(){ awk -F, -v c="$1" -v h="$2" 'NR==1{for(i=1;i<=NF;i++)k[$i]=i;next} $1==c{print $k[h]}' "$OUT"; }

# 1. budget_stop sorts before none_down. The rule stops this cell on its
#    own numbers, so a reader must see the rule, not the box release.
check "ncpc_alignT branch is the rule's"   "none_down" "$(col arm6_v2_ncpc_alignT rule_branch)"
check "ncpc_alignT ended by the rule"      "rule"      "$(col arm6_v2_ncpc_alignT ended_by)"
check "ncpc_alignT keeps the budget row"   "budget_stop none_down" "$(col arm6_v2_ncpc_alignT recorded)"

# 2. The opposite error: budget_stop sorts FIRST here too, but the rule
#    says extend, so this cell is a budget stop and must read as one.
check "alignS branch is one_down"          "one_down"  "$(col arm5_combab_alignS rule_branch)"
check "alignS ended by the budget"         "budget"    "$(col arm5_combab_alignS ended_by)"
check "alignS carries only its down head"  "student"   "$(col arm5_combab_alignS heads_next)"

# 3. A ceiling park is not a budget park.
check "alignT branch is both_down"         "both_down" "$(col arm5_combab_alignT rule_branch)"
check "alignT ended by the session"        "session"   "$(col arm5_combab_alignT ended_by)"

# 4. A live driver outranks both park rows.
python3 "$HERE/stop_reason.py" --ladder "$LADDER" --decisions "$DEC" \
        --out "$OUT.live" >/dev/null 2>&1
LIVE_OUT="$OUT" ; OUT="$OUT.live"
if pgrep -f "[l]adder\.py --cells.*arm5_combab_alignT" >/dev/null 2>&1; then
  check "alignT reads running while its driver is up" "running" "$(col arm5_combab_alignT ended_by)"
else
  ok "alignT has no live driver here — probe case not exercised"
fi
OUT="$LIVE_OUT"

# 5. The 40k stop extends unconditionally, and a cell that never reached
#    100k must not be resolved against a stop it has no scores for.
check "arm1_nse resolves at its furthest scored stop" "40000" "$(col arm1_nse last_stop)"
check "arm1_nse branch is unconditional" "unconditional" "$(col arm1_nse rule_branch)"
check "arm1_nse has no park row, so it is open" "open" "$(col arm1_nse ended_by)"

# 6. Every emitted branch is one a recorded row agrees with. Reading the
#    LAST field would now read probed_at, so the column is resolved by name.
n_no=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)k[$i]=i;next} $k["rule_in_recorded"]=="no"' "$OUT" | wc -l)
check "every row agrees with a recorded branch" "0" "$n_no"

# 6b. `rule_in_recorded` says "some row agrees"; `stale_rows` says "and
#     these do not". The five parks are exactly what the old single flag
#     read `yes` through, so each has to be named here.
check "alignS names its stale budget row"  "budget_stop" "$(col arm5_combab_alignS stale_rows)"
check "alignT names its stale ceiling row" "session_end" "$(col arm5_combab_alignT stale_rows)"
check "ncpc_alignT names its stale row"    "budget_stop" "$(col arm6_v2_ncpc_alignT stale_rows)"
check "a clean cell names nothing stale"   ""            "$(col arm1_nse stale_rows)"

# 6c. --no-probe leaves probed_at empty, which is what makes the file
#     reproducible from the CSVs alone.
check "no-probe leaves probed_at empty" "" "$(col arm1_nse probed_at)"

# 7. A cell with no score at all is left out rather than guessed at.
check "only scored cells are emitted" "4" "$(( $(wc -l <"$OUT") - 1 ))"

# 8. Schema.
check "header" \
  "cell,last_stop,rule_branch,extend,heads_next,ended_by,ended_by_evidence,recorded,stale_rows,rule_in_recorded,probed_at" \
  "$(head -1 "$OUT")"

# 9. `running` is the one value a live probe produces, so it has to leave a
#    path behind that outlives the process it read.
check "a budget park points at the file that records it" \
  "decisions_all.csv branch=budget_stop" "$(col arm5_combab_alignS ended_by_evidence)"
check "a rule stop needs no evidence beyond the scores" \
  "" "$(col arm6_v2_ncpc_alignT ended_by_evidence)"
check "no CR in the output" "0" "$(tr -cd '\r' <"$OUT" | wc -c)"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
