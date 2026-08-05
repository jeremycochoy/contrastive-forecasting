#!/bin/bash
# #393 — guard on scripts/merge_pooled.sh.
#
# Usage:  bash scripts/test_merge_pooled.sh
#
# The pooling bug this guards against loses rows without failing: the output
# is a valid CSV, sorted, just short. Case 1 is the exact row pair that went
# missing — one stop, two heads — and it is the reason this file exists.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE="$HERE/merge_pooled.sh"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0

ok(){   printf '  ok   %s\n' "$1"; pass=$((pass+1)); }
bad(){  printf '  FAIL %s\n' "$1"; printf '       %s\n' "$2"; fail=$((fail+1)); }
check(){ [ "$2" = "$3" ] && ok "$1" || bad "$1" "want [$3], got [$2]"; }

fixture(){  # <case> <basename> <machine> <lines...>
  local res="$TMP/$1" base="$2" m="$3"; shift 3
  mkdir -p "$res/per_machine"
  printf '%s\n' "$@" > "$res/per_machine/${base}_${m}.csv"
  echo "$res"
}
rows(){ tail -n +2 "$1" ; }

LH='cell,arm,align,stop,head,head_steps,ema_tau,gm_rel_mase'
DH='cell,stop,branch,extend,heads_next'

# --- 1. two rows differing only in `head` both survive ---------------------
# This is the defect. `sort -u -t, -k1,1 -k4,4n` keyed on (cell, stop) and
# dropped whichever head sorted second, so every teacher score vanished.
R=$(fixture heads ladder elisa "$LH" \
  'c1,a,student,40000,student,15000,0.940000,1.160300' \
  'c1,a,student,40000,teacher,15000,0.940000,1.154400')
RES="$R" bash "$MERGE" ladder >/dev/null
check "both heads kept at one stop"        "$(rows "$R/ladder_all.csv" | wc -l)" "2"
check "  student row present"              "$(grep -c ',student,15000,' "$R/ladder_all.csv")" "1"
check "  teacher row present"              "$(grep -c ',teacher,15000,' "$R/ladder_all.csv")" "1"

# --- 2. two stops of one cell both survive in decisions --------------------
# Same class of bug on the other file: its old key was (cell, extend), and
# extend is 1 on every unconditional stop, so the second stop of a cell would
# have replaced the first.
R=$(fixture stops decisions elisa "$DH" \
  'c1,40000,unconditional,1,student teacher' \
  'c1,100000,unconditional,1,student teacher')
RES="$R" bash "$MERGE" decisions >/dev/null
check "both stops kept for one cell"       "$(rows "$R/decisions_all.csv" | wc -l)" "2"

# --- 3. stops sort numerically, not as text -------------------------------
R=$(fixture order ladder elisa "$LH" \
  'c1,a,student,100000,student,30000,1.000000,1.100000' \
  'c1,a,student,40000,student,15000,0.940000,1.200000')
RES="$R" bash "$MERGE" ladder >/dev/null
check "40000 sorts before 100000"          "$(rows "$R/ladder_all.csv" | head -1 | cut -d, -f4)" "40000"

# --- 4. the same row on two machines collapses to one ---------------------
# Machines are copied verbatim and a cell can be visible from two of them.
R=$(fixture dedup ladder elisa "$LH" \
  'c1,a,student,40000,student,15000,0.940000,1.160300')
printf '%s\n%s\n' "$LH" 'c1,a,student,40000,student,15000,0.940000,1.160300' \
  > "$R/per_machine/ladder_vastA.csv"
RES="$R" bash "$MERGE" ladder >/dev/null
check "identical row across machines dedupes" "$(rows "$R/ladder_all.csv" | wc -l)" "1"

# --- 5. one key, two different scores: keep both and say so ---------------
R=$(fixture rescore ladder elisa "$LH" \
  'c1,a,student,40000,student,15000,0.940000,1.160300')
printf '%s\n%s\n' "$LH" 'c1,a,student,40000,student,15000,0.940000,1.999900' \
  > "$R/per_machine/ladder_vastA.csv"
out=$(RES="$R" bash "$MERGE" ladder 2>&1)
check "conflicting scores both kept"       "$(rows "$R/ladder_all.csv" | wc -l)" "2"
check "  conflict reported"                "$(grep -c 'WARN' <<<"$out")" "1"

# --- 6. a moved column moves the key ---------------------------------------
# The key is declared by name, so this header still pools on head.
R=$(fixture renamed ladder elisa 'cell,arm,align,run_id,stop,head,head_steps,ema_tau,gm_rel_mase' \
  'c1,a,student,r0,40000,student,15000,0.940000,1.160300' \
  'c1,a,student,r0,40000,teacher,15000,0.940000,1.154400')
RES="$R" bash "$MERGE" ladder >/dev/null
check "key follows an inserted column"     "$(rows "$R/ladder_all.csv" | wc -l)" "2"

# --- 7. a header that lacks a key column aborts, loudly --------------------
R=$(fixture nokey ladder elisa 'cell,arm,align,stop,head_steps,ema_tau,gm_rel_mase' \
  'c1,a,student,40000,15000,0.940000,1.160300')
out=$(RES="$R" bash "$MERGE" ladder 2>&1); rc=$?
check "missing key column aborts"          "$rc" "5"
check "  and names the column"             "$(grep -c "column 'head'" <<<"$out")" "1"

# --- 8. machines disagreeing on the header aborts --------------------------
R=$(fixture badhdr ladder elisa "$LH" \
  'c1,a,student,40000,student,15000,0.940000,1.160300')
printf '%s\n%s\n' 'cell,stop,gm_rel_mase' 'c1,40000,1.1' \
  > "$R/per_machine/ladder_vastA.csv"
RES="$R" bash "$MERGE" ladder >/dev/null 2>&1; rc=$?
check "header mismatch aborts"             "$rc" "4"

# --- 9. no input is not an error -------------------------------------------
R="$TMP/empty"; mkdir -p "$R/per_machine"
RES="$R" bash "$MERGE" ladder >/dev/null 2>&1
check "empty input exits clean"            "$?" "0"
check "  and writes no pooled file"        "$([ -e "$R/ladder_all.csv" ] && echo yes || echo no)" "no"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
