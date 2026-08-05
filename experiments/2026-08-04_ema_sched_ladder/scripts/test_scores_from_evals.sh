#!/bin/bash
# #393 — guard on scripts/scores_from_evals.py.
#
# Usage:  bash scripts/test_scores_from_evals.sh
#
# What this protects is the claim that a score survives the machine that
# asked for it. Every number in this card is computed on elisa, but until
# 08-05 the only path from a score to the pooled table ran through the
# BOX's ladder.csv — and ladder.py appends a stop's two rows only after
# both heads return, so a stop with one head scored and one still
# evaluating contributed nothing. Release that box and a measurement that
# cost a 100k-step backbone and a 30k-step head leaves no trace, silently:
# the pooled table is still valid CSV, just short.
#
# Case 1 is the row that was actually missing when this was written —
# arm6_v2_ncpc_alignT bb100k student, measured 1.3904 at 16:12, absent
# from ladder_all.csv at 18:40.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$HERE/scores_from_evals.py"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

ok(){   printf '  ok   %s\n' "$1"; pass=$((pass+1)); }
bad(){  printf '  FAIL %s\n' "$1"; printf '       %s\n' "$2"; fail=$((fail+1)); }
check(){ [ "$2" = "$3" ] && ok "$1" || bad "$1" "want [$3], got [$2]"; }

# A finished eval on disk: the score, and the 97-config table behind it.
plant(){  # <stop dir> <score> [config rows]
  local d="$1" n="${3:-97}" i
  mkdir -p "$d/gift"
  { echo "dataset,metric,value"
    for (( i = 0; i < n; i++ )); do echo "cfg$i,MASE,1.0"; done
  } > "$d/gift/all_results.csv"
  printf '%s\n' "$2" > "$d/score.txt"
}

runs(){ echo "$TMP/$1"; }
gen(){ python3 "$GEN" --runs "$1" --check 2>/dev/null; }

# --- 1. a broker score whose box never wrote a ladder row ------------------
R=$(runs one)
plant "$R/_broker/g/arm6_v2_ncpc_alignT/bb100k_student" 1.3904
OUT=$(gen "$R")
check "the orphaned broker score is recovered" \
      "$(grep -c '^arm6_v2_ncpc_alignT,arm6_v2_ncpc,teacher,100000,student,30000,1.000000,1.390400$' <<<"$OUT")" "1"
check "  and nothing else"                     "$(tail -n +2 <<<"$OUT" | wc -l)" "1"

# --- 2. both heads of one stop, from two different layouts -----------------
# A cell that trains on elisa writes its score one level up from the stop
# directory; a cell on a box gets score.txt inside the broker's working
# copy. Both are the same measurement and both have to appear.
R=$(runs two)
plant "$R/_broker/a/arm5_combab_alignS/bb100k_teacher" 1.2347
mkdir -p "$R/arm5_combab_alignS/eval/bb100k_student/gift"
plant "$R/arm5_combab_alignS/eval/bb100k_student" 1.2102
mv "$R/arm5_combab_alignS/eval/bb100k_student/score.txt" \
   "$R/arm5_combab_alignS/eval/score_bb100k_student.txt"
OUT=$(gen "$R")
check "both heads of one stop are emitted"     "$(tail -n +2 <<<"$OUT" | wc -l)" "2"
check "  student from the elisa layout"        "$(grep -c ',100000,student,30000,1.000000,1.210200$' <<<"$OUT")" "1"
check "  teacher from the broker layout"       "$(grep -c ',100000,teacher,30000,1.000000,1.234700$' <<<"$OUT")" "1"

# --- 3. a score with no 97-config table behind it is refused ---------------
# eval_local.sh writes the score last, so a partial eval leaves none. A
# truncated pull or a half-copied directory could still leave a number
# with nothing under it, and a number with no evidence is not a result.
R=$(runs short)
plant "$R/_broker/a/arm1_nse/bb40k_student" 1.4347 96
OUT=$(gen "$R")
check "a 96-config eval is refused"            "$(tail -n +2 <<<"$OUT" | wc -l)" "0"

R=$(runs noevidence)
mkdir -p "$R/_broker/a/arm1_nse/bb40k_student"
printf '1.4347\n' > "$R/_broker/a/arm1_nse/bb40k_student/score.txt"
OUT=$(gen "$R")
check "a score with no all_results.csv is refused" "$(tail -n +2 <<<"$OUT" | wc -l)" "0"

# --- 4. the head budget and alpha follow the stop, not the file ------------
# 15,000 steps at bb40k and 30,000 from bb100k on; alpha 0.94 at 40k and
# 1.00 at 100k, anchored to the fixed step and not to the leg budget.
R=$(runs budgets)
plant "$R/_broker/a/arm4_combab/bb40k_teacher" 1.287
plant "$R/_broker/a/arm4_combab/bb100k_teacher" 1.1
OUT=$(gen "$R")
check "bb40k carries 15000 steps at alpha 0.94" \
      "$(grep -c ',40000,teacher,15000,0.940000,' <<<"$OUT")" "1"
check "bb100k carries 30000 steps at alpha 1.00" \
      "$(grep -c ',100000,teacher,30000,1.000000,' <<<"$OUT")" "1"

# --- 5. a directory that is not a cell of this card is ignored -------------
R=$(runs stranger)
plant "$R/_broker/a/some_other_run/bb100k_student" 1.0
OUT=$(gen "$R")
check "an unknown cell contributes no row"     "$(tail -n +2 <<<"$OUT" | wc -l)" "0"

# --- 6. the output is exactly the pooled schema ----------------------------
# merge_pooled.sh aborts if one per-machine file disagrees with the others
# on its header, so this file has to match ladder.csv column for column.
R=$(runs schema)
plant "$R/_broker/a/arm1_nse/bb40k_student" 1.4347
check "header matches ladder.csv" "$(gen "$R" | head -1)" \
      "cell,arm,align,stop,head,head_steps,ema_tau,gm_rel_mase"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
