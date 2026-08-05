#!/bin/bash
# Mutation check for the #390 half of tests/test_eval_checkpoint_resolution.py.
#
# A test that passes against a broken script proves nothing. This breaks the
# two call sites one way at a time and requires the named tests to fail. Run
# from anywhere:
#
#   bash experiments/2026-08-01_lalign_teacher/scripts/mutation_check_ckpt_resolution.sh
#
# Exits 0 when every mutation was caught, non-zero otherwise. Both scripts are
# copied aside before the first mutation and restored after each one, so an
# interrupted run leaves the checkout as it found it.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2

EVAL="experiments/2026-08-01_lalign_teacher/scripts/eval_arm.sh"
PIPE="experiments/2026-08-01_lalign_teacher/scripts/pipeline.sh"
TESTS="tests/test_eval_checkpoint_resolution.py"

SAFE="$(mktemp -d)"
cp "$EVAL" "$SAFE/eval_arm.sh"
cp "$PIPE" "$SAFE/pipeline.sh"
restore(){ cp "$SAFE/eval_arm.sh" "$EVAL"; cp "$SAFE/pipeline.sh" "$PIPE"; }
trap 'restore; rm -rf "$SAFE"' EXIT

# The two halves of a mutation are joined by \x01 — no shell script contains
# it, and unlike NUL a shell variable can hold it.
SEP=$'\x01'
fails=0

apply(){  # file  "<old>\x01<new>"
  python3 - "$1" "$2" <<'PY'
import sys
path, prog = sys.argv[1], sys.argv[2]
old, new = prog.split("\x01")
src = open(path).read()
if old not in src:
    sys.exit(f"mutation target not found in {path}:\n{old}")
open(path, "w").write(src.replace(old, new, 1))
PY
}

mutate(){  # <label> <file> <old\x01new> <test>...
  local label="$1" file="$2" prog="$3"; shift 3
  restore
  if ! apply "$file" "$prog"; then
    echo "ERROR    $label — could not apply the mutation"
    fails=$((fails + 1)); return
  fi
  local expr="" t
  for t in "$@"; do expr="${expr:+$expr or }$t"; done
  local out
  out=$(python3 -m pytest "$TESTS" -q --no-header -k "$expr" 2>&1)
  local n_sel; n_sel=$(echo "$out" | grep -oE '[0-9]+ (passed|failed)' | head -1)
  if echo "$out" | grep -qE '[0-9]+ failed'; then
    echo "caught   $label"
  else
    echo "SURVIVED $label — [$expr] still passes ($n_sel)"
    fails=$((fails + 1))
  fi
}

# --- eval_arm.sh ----------------------------------------------------------
mutate "eval: mtime pick restored" "$EVAL" \
"BB=\$(bash \"\$CKPT_RESOLVER\" \"\$RUNS\" \"\$NAME\" \"\$BB_STEP_K\" \"\$BB_CHECKPOINT\") || exit \$?${SEP}BB=\$(ls -t \"\$RUNS/\${NAME}\"_\${BB_STEP_K}k.pth \"\$RUNS/\${NAME}\"_r*_\${BB_STEP_K}k.pth 2>/dev/null | head -1)" \
  test_390_eval_arm_has_no_mtime_pick test_390_eval_arm_calls_the_resolver \
  test_390_eval_arm_aborts_on_two_candidates

mutate "eval: resolver failure ignored" "$EVAL" \
"\"\$BB_CHECKPOINT\") || exit \$?${SEP}\"\$BB_CHECKPOINT\")" \
  test_390_eval_arm_aborts_when_resolution_fails \
  test_390_eval_arm_aborts_on_two_candidates

mutate "eval: resolver taken from \$WT" "$EVAL" \
"CKPT_RESOLVER=\"\$(cd \"\$HERE/../../..\" && pwd)/scripts${SEP}CKPT_RESOLVER=\"\$(cd \"\$WT\" && pwd)/scripts" \
  test_390_eval_arm_takes_the_resolver_from_its_own_checkout \
  test_390_eval_arm_resolves_with_a_wt_that_has_no_resolver

mutate "eval: missing-resolver guard dropped" "$EVAL" \
"[ -f \"\$CKPT_RESOLVER\" ] || { echo \"ABORT: no checkpoint resolver at \$CKPT_RESOLVER\" >&2; exit 2; }${SEP}: no guard" \
  test_390_eval_arm_guards_a_missing_resolver

mutate "eval: override not passed through" "$EVAL" \
"\"\$BB_STEP_K\" \"\$BB_CHECKPOINT\"${SEP}\"\$BB_STEP_K\"" \
  test_390_eval_arm_calls_the_resolver \
  test_390_eval_arm_override_selects_the_named_replicate

mutate "eval: override knob removed" "$EVAL" \
"BB_CHECKPOINT=\"\${BB_CHECKPOINT:-}\"${SEP}BB_CHECKPOINT=\"\"" \
  test_390_eval_arm_exposes_the_override

mutate "eval: override undocumented" "$EVAL" \
"[BB_CHECKPOINT=<path>] bash eval_arm.sh${SEP}bash eval_arm.sh" \
  test_390_eval_arm_exposes_the_override

mutate "eval: logs the basename again" "$EVAL" \
"say \"start on GPU \$BB_GPU (backbone=\$BB)\"${SEP}say \"start on GPU \$BB_GPU (backbone=\$(basename \"\$BB\"))\"" \
  test_390_eval_arm_logs_the_resolved_path \
  test_390_eval_arm_records_the_resolved_path_in_its_log \
  test_390_eval_arm_override_selects_the_named_replicate \
  test_390_eval_arm_resolves_with_a_wt_that_has_no_resolver

# --- pipeline.sh ----------------------------------------------------------
mutate "pipeline: mtime pick restored" "$PIPE" \
"hit=\$(bash \"\$CKPT_RESOLVER\" \"\$RUNS\" \"\$name\" \"\$k\" 2>\"\$errf\")
    rc=\$?${SEP}hit=\$(ls -t \"\$RUNS/\${name}\"_\${k}k.pth \"\$RUNS/\${name}\"_r*_\${k}k.pth 2>/dev/null | head -1)
    rc=0; [ -n \"\$hit\" ] || rc=3" \
  test_390_pipeline_has_no_mtime_pick test_390_pipeline_calls_the_resolver \
  test_390_pipeline_hands_only_unambiguous_arms_downstream

mutate "pipeline: ambiguous arm kept" "$PIPE" \
"      *) log \"  DROP \$arm${SEP}      *) out+=(\"\$arm\"); log \"  DROP \$arm" \
  test_390_pipeline_hands_only_unambiguous_arms_downstream

mutate "pipeline: READY log on stdout" "$PIPE" \
"log \"  READY \$arm — \$hit\" >&2${SEP}log \"  READY \$arm — \$hit\"" \
  test_390_pipeline_arm_list_stays_on_stdout

mutate "pipeline: resolved path not logged" "$PIPE" \
"log \"  READY \$arm — \$hit\" >&2${SEP}log \"  READY \$arm\" >&2" \
  test_390_pipeline_records_the_resolved_path

mutate "pipeline: candidates not listed" "$PIPE" \
"         while IFS= read -r line; do log \"    \$line\" >&2; done <\"\$errf\" ;;${SEP}         ;;" \
  test_390_pipeline_drops_an_ambiguous_arm_naming_both_candidates

mutate "pipeline: resolver taken from \$WT" "$PIPE" \
"CKPT_RESOLVER=\"\$(cd \"\$HERE/../../..\" && pwd)/scripts${SEP}CKPT_RESOLVER=\"\$(cd \"\$WT\" && pwd)/scripts" \
  test_390_pipeline_takes_the_resolver_from_its_own_checkout

mutate "pipeline: missing-resolver guard dropped" "$PIPE" \
"[ -f \"\$CKPT_RESOLVER\" ] || { echo \"ABORT: no checkpoint resolver at \$CKPT_RESOLVER\" >&2; exit 2; }${SEP}: no guard" \
  test_390_pipeline_guards_a_missing_resolver

restore
echo
if [ "$fails" -eq 0 ]; then
  echo "ALL MUTATIONS CAUGHT"
else
  echo "$fails MUTATION(S) SURVIVED"
fi
exit "$fails"
