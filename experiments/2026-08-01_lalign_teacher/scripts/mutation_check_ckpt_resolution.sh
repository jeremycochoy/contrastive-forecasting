#!/bin/bash
# Mutation check for the #390 half of tests/test_eval_checkpoint_resolution.py.
#
# A test that passes against a broken script proves nothing. This breaks the
# call sites and the resolver one way at a time and requires the named tests
# to fail. Run from anywhere:
#
#   bash experiments/2026-08-01_lalign_teacher/scripts/mutation_check_ckpt_resolution.sh
#
# Exits 0 when every mutation was caught, non-zero otherwise. Every script is
# copied aside before the first mutation and restored after each one, so an
# interrupted run leaves the checkout as it found it.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || exit 2

RESOLVER="scripts/resolve_eval_checkpoint.sh"
IDENTITY="scripts/eval_cell_identity.sh"
IDENTITY_PY="scripts/eval_cell_identity.py"
EVAL="experiments/2026-08-01_lalign_teacher/scripts/eval_arm.sh"
EVAL379="experiments/2026-07-21_split_pred_rep_small/scripts/eval_2L_gm_mase.sh"
PIPE="experiments/2026-08-01_lalign_teacher/scripts/pipeline.sh"
WAVE="experiments/2026-08-01_lalign_teacher/scripts/eval_wave.sh"
GATE="experiments/2026-08-01_lalign_teacher/scripts/select_wave3.py"
BOOT="experiments/2026-08-01_lalign_teacher/scripts/eval_bootstrap.py"
TABLE="experiments/2026-08-01_lalign_teacher/scripts/make_gm_table.py"
BATCH="experiments/2026-08-01_lalign_teacher/scripts/run_student_control_batch.sh"
CELLS="reports/2026-08-04_lalign_teacher/plots/_cells.py"
RUNARM="experiments/2026-08-01_lalign_teacher/scripts/run_arm.sh"
PARITY="experiments/2026-08-01_lalign_teacher/scripts/verify_ckpt_resolution_parity.sh"
TESTS="tests/test_eval_checkpoint_resolution.py tests/test_eval_cell_identity.py
       tests/test_390_analysis_scripts.py tests/test_390_eval_path_shape.py"

FILES=("$RESOLVER" "$IDENTITY" "$IDENTITY_PY" "$EVAL" "$EVAL379" "$PIPE"
       "$WAVE" "$GATE" "$BOOT" "$TABLE" "$BATCH" "$CELLS" "$RUNARM" "$PARITY")
SAFE="$(mktemp -d)"
for f in "${FILES[@]}"; do mkdir -p "$SAFE/$(dirname "$f")"; cp "$f" "$SAFE/$f"; done
restore(){ local f; for f in "${FILES[@]}"; do cp "$SAFE/$f" "$f"; done; }
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
  # shellcheck disable=SC2086  # $TESTS is a list of test files
  out=$(python3 -m pytest $TESTS -q --no-header -k "$expr" 2>&1)
  local n_sel; n_sel=$(echo "$out" | grep -oE '[0-9]+ (passed|failed)' | head -1)
  if echo "$out" | grep -qE '[0-9]+ failed'; then
    echo "caught   $label"
  else
    echo "SURVIVED $label — [$expr] still passes ($n_sel)"
    fails=$((fails + 1))
  fi
}

# --- resolve_eval_checkpoint.sh ------------------------------------------
# Single-quoted halves: these mutations are regexes, and one lost backslash
# would silently mutate something else.
mutate "resolver: override not checked at all" "$IDENTITY" \
'  [[ "$(basename "$3")" =~ ^"$1"(_r[0-9]+)?_"$2"k\.pth$ ]]'"$SEP"'  [[ "$(basename "$3")" =~ ^.*$ ]]' \
  test_override_from_another_step_aborts test_override_from_another_run_aborts \
  test_override_rejects_the_optimizer_sidecar \
  test_390_eval_arm_rejects_an_override_from_another_step

mutate "resolver: override step not pinned" "$IDENTITY" \
'  [[ "$(basename "$3")" =~ ^"$1"(_r[0-9]+)?_"$2"k\.pth$ ]]'"$SEP"'  [[ "$(basename "$3")" =~ ^"$1"(_r[0-9]+)?_[0-9]+k\.pth$ ]]' \
  test_override_from_another_step_aborts \
  test_390_eval_arm_rejects_an_override_from_another_step

mutate "resolver: override run name not pinned" "$IDENTITY" \
'  [[ "$(basename "$3")" =~ ^"$1"(_r[0-9]+)?_"$2"k\.pth$ ]]'"$SEP"'  [[ "$(basename "$3")" =~ ^.*_"$2"k\.pth$ ]]' \
  test_override_from_another_run_aborts

mutate "resolver: the replicate form stops being accepted" "$IDENTITY" \
'  [[ "$(basename "$3")" =~ ^"$1"(_r[0-9]+)?_"$2"k\.pth$ ]]'"$SEP"'  [[ "$(basename "$3")" =~ ^"$1"_"$2"k\.pth$ ]]' \
  test_explicit_override_selects_the_named_file \
  test_override_accepts_a_two_digit_replicate \
  test_390_eval_arm_override_selects_the_named_replicate

mutate "resolver: loose _r*_ candidate glob restored" "$RESOLVER" \
'for f in "$RUNS/${NAME}_${STEP_K}k.pth" "$RUNS/${NAME}"_r[0-9]*_${STEP_K}k.pth; do'"$SEP"'for f in "$RUNS/${NAME}_${STEP_K}k.pth" "$RUNS/${NAME}"_r*_${STEP_K}k.pth; do' \
  test_a_non_replicate_r_suffix_is_not_a_candidate \
  test_a_non_replicate_r_suffix_does_not_create_an_ambiguity

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
'ROOT="$(cd -P "$HERE/../../.." && pwd)"'"$SEP"'ROOT="$WT"' \
  test_390_eval_arm_takes_the_resolver_from_its_own_checkout \
  test_390_eval_arm_resolves_with_a_wt_that_has_no_resolver

mutate "eval: script path resolved logically" "$EVAL" \
'HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -P "$HERE/../../.." && pwd)"'"$SEP"'HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"' \
  test_390_eval_arm_takes_the_resolver_from_its_own_checkout \
  test_390_eval_arm_finds_its_own_resolver_through_a_symlink

mutate "eval: missing-resolver guard dropped" "$EVAL" \
'[ -f "$CKPT_RESOLVER" ] || { echo "ABORT: no checkpoint resolver at $CKPT_RESOLVER" >&2; exit $E_SETUP; }'"$SEP"': no guard' \
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
"      5) log \"  DROP \$arm${SEP}      5) out+=(\"\$arm\"); log \"  DROP \$arm" \
  test_390_pipeline_hands_only_unambiguous_arms_downstream

mutate "pipeline: arm kept after a resolver failure" "$PIPE" \
"      *) log \"  DROP \$arm${SEP}      *) out+=(\"\$arm\"); log \"  DROP \$arm" \
  test_390_pipeline_reports_an_unexpected_resolver_failure

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
'ROOT="$(cd -P "$HERE/../../.." && pwd)"'"$SEP"'ROOT="$WT"' \
  test_390_pipeline_takes_the_resolver_from_its_own_checkout

mutate "pipeline: script path resolved logically" "$PIPE" \
'HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -P "$HERE/../../.." && pwd)"'"$SEP"'HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"' \
  test_390_pipeline_takes_the_resolver_from_its_own_checkout \
  test_390_pipeline_finds_its_own_resolver_through_a_symlink

mutate "pipeline: missing-resolver guard dropped" "$PIPE" \
"[ -f \"\$CKPT_RESOLVER\" ] || { echo \"ABORT: no checkpoint resolver at \$CKPT_RESOLVER\" >&2; exit 2; }${SEP}: no guard" \
  test_390_pipeline_guards_a_missing_resolver

mutate "pipeline: any failure reported as an ambiguity" "$PIPE" \
'      *) log "  DROP $arm — checkpoint resolver failed rc=$rc:" >&2'"$SEP"'      *) log "  DROP $arm — ${k}k backbone is ambiguous, not choosing:" >&2' \
  test_390_pipeline_reports_an_unexpected_resolver_failure

mutate "pipeline: ambiguity folded back into the catch-all" "$PIPE" \
'      5) log "  DROP $arm — ${k}k backbone is ambiguous, not choosing:" >&2'"$SEP"'      4) log "  DROP $arm — ${k}k backbone is ambiguous, not choosing:" >&2' \
  test_390_pipeline_matches_the_ambiguous_code_explicitly \
  test_390_pipeline_drops_an_ambiguous_arm_naming_both_candidates

# --- run_arm.sh: the `_FINAL.pth` fallback -------------------------------
mutate "run_arm: mtime pick restored in the FINAL fallback" "$RUNARM" \
'  snap=$(bash "$CKPT_RESOLVER" "$RUNS" "$NAME" "$last_k" 2>"$errf"); rc=$?'"$SEP"'  snap=$(ls -t "$RUNS/${NAME}"_*k.pth 2>/dev/null | grep -v optimizer | head -1); rc=0' \
  test_run_arm_final_fallback_aborts_on_two_candidates

mutate "run_arm: resolver taken from \$WT" "$RUNARM" \
'CKPT_RESOLVER="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts'"$SEP"'CKPT_RESOLVER="$(cd "$WT" && pwd)/scripts' \
  test_run_arm_takes_the_resolver_from_its_own_checkout \
  test_run_arm_final_fallback_copies_the_one_snapshot

mutate "run_arm: script path resolved logically" "$RUNARM" \
'CKPT_RESOLVER="$(cd -P "$(dirname'"$SEP"'CKPT_RESOLVER="$(cd "$(dirname' \
  test_run_arm_finds_the_resolver_through_a_symlinked_scripts_dir

# --- verify_ckpt_resolution_parity.sh: what the run claims ---------------
mutate "parity: multi-candidate pairs not counted" "$PARITY" \
'  [ "$cand" -gt 1 ] && multi=$((multi + 1))'"$SEP"'  :' \
  test_parity_counts_a_real_ambiguity

mutate "parity: multi-candidate count not reported" "$PARITY" \
'echo "multi-candidate pairs : $multi"'"$SEP"':' \
  test_parity_counts_the_pairs_that_could_be_ambiguous \
  test_parity_counts_a_real_ambiguity

mutate "parity: scope note dropped" "$PARITY" \
'  echo "SCOPE   : every pair on disk has exactly one candidate, so this run does not"'"$SEP"'  echo "SCOPE   : parity holds."' \
  test_parity_states_what_it_does_not_show

# --- eval_cell_identity.sh: the output identity --------------------------
mutate "identity: cell name drops the replicate token" "$IDENTITY" \
"  printf '%s%s_bb%sk%s_hd%ss' \"\$1\" \"\$(head_seed_tag \"\$seed\")\" \"\$2\" \"\$3\" \"\$4\"${SEP}  printf '%s%s_bb%sk_hd%ss' \"\$1\" \"\$(head_seed_tag \"\$seed\")\" \"\$2\" \"\$4\"" \
  test_cell_name_carries_the_replicate_beside_its_step \
  test_a_replicate_cell_is_not_the_base_cell \
  test_390_eval_arm_replicate_gets_its_own_cell \
  test_390_eval_arm_replicate_does_not_lift_the_base_runs_number

mutate "identity: cell name drops the head-seed token" "$IDENTITY" \
"  printf '%s%s_bb%sk%s_hd%ss' \"\$1\" \"\$(head_seed_tag \"\$seed\")\" \"\$2\" \"\$3\" \"\$4\"${SEP}  printf '%s_bb%sk%s_hd%ss' \"\$1\" \"\$2\" \"\$3\" \"\$4\"" \
  test_cell_name_carries_the_head_seed_beside_the_slug \
  test_a_second_head_seed_is_not_the_first_seeds_cell \
  test_390_eval_arm_head_seed_gets_its_own_cell \
  test_390_eval_arm_head_seed_does_not_lift_the_other_seeds_number

mutate "identity: every head seed carries a token, so the base cells move" "$IDENTITY" \
'  [ "$1" = "$EVAL_DEFAULT_HEAD_SEED" ] || printf '"'"'_s%s'"'"' "$1"'"$SEP"'  printf '"'"'_s%s'"'"' "$1"' \
  test_the_wave_seed_has_no_token \
  test_cell_name_without_a_replicate_is_the_committed_shape \
  test_390_eval_arm_base_cell_keeps_its_committed_name

mutate "identity: the head seed is read inside the substitution again" "$IDENTITY" \
'  local seed="$5"
  printf'"$SEP"'  local seed="${5:-$EVAL_DEFAULT_HEAD_SEED}"
  printf' \
  test_the_head_seed_is_not_optional

mutate "identity: an unrecognised path reads as the base run" "$IDENTITY" \
'  ckpt_is_run_step "$@" || return 1'"$SEP"'  ckpt_is_run_step "$@" || return 0' \
  test_a_path_of_another_pair_has_no_token

mutate "identity: cell lookup misses the replicate cells" "$IDENTITY" \
'  for f in "$1/$(eval_cell_name "$2" "$3" "" "$4" "$seed")_summary.txt" \
           "$1/${slug}_bb$3k"_r[0-9]*"_hd$4s_summary.txt"; do'"$SEP"'  for f in "$1/$(eval_cell_name "$2" "$3" "" "$4" "$seed")_summary.txt"; do' \
  test_a_replicate_cell_summary_is_found \
  test_both_replicates_are_returned_not_one

mutate "identity: the replicate half of the lookup ignores the head seed" "$IDENTITY" \
'  slug="$2$(head_seed_tag "$seed")"'"$SEP"'  slug="$2"' \
  test_a_replicate_cell_of_another_seed_is_not_this_cells_summary \
  test_a_head_seed_cell_measured_on_a_replicate_is_found \
  test_python_and_bash_find_the_same_summaries

mutate "identity: loose _r*_ in the cell-summary glob" "$IDENTITY" \
'"$1/${slug}_bb$3k"_r[0-9]*"_hd$4s_summary.txt"'"$SEP"'"$1/${slug}_bb$3k"_r*"_hd$4s_summary.txt"' \
  test_a_non_replicate_suffix_is_not_a_replicate_cell

mutate "identity: the bad-tag code goes back to being per-script" "$IDENTITY" \
'E_BAD_TAG=25'"$SEP"'E_BAD_TAG=20' \
  test_the_bad_tag_code_is_the_librarys_and_both_evals_use_it

# --- eval_cell_identity.py: the same grammar, the other binding ----------
# The readers are Python. A drift between the two bindings is a reader that
# looks for a name nothing writes.
mutate "identity.py: the Python default head seed drifts" "$IDENTITY_PY" \
'DEFAULT_HEAD_SEED = "20260722"'"$SEP"'DEFAULT_HEAD_SEED = "20260721"' \
  test_python_and_bash_agree_on_the_default_head_seed \
  test_python_and_bash_build_the_same_cell_name \
  test_python_and_bash_agree_on_the_head_seed_token

mutate "identity.py: the Python cell name drops the replicate" "$IDENTITY_PY" \
'            f"_bb{int(bb_k)}k{replicate}_hd{int(head_steps)}s")'"$SEP"'            f"_bb{int(bb_k)}k_hd{int(head_steps)}s")' \
  test_python_and_bash_build_the_same_cell_name \
  test_python_parses_back_what_it_builds

mutate "identity.py: the Python lookup misses the replicate cells" "$IDENTITY_PY" \
'        + r"(?P<repl>_r\d+)?"'"$SEP"'        + r""' \
  test_python_and_bash_find_the_same_summaries \
  test_the_figures_find_a_replicate_backed_cell \
  test_the_bootstrap_finds_a_replicate_backed_cell \
  test_the_wave3_gate_finds_a_replicate_backed_cell

mutate "identity.py: loose _r.* in the Python cell pattern" "$IDENTITY_PY" \
'        + r"(?P<repl>_r\d+)?"'"$SEP"'        + r"(?P<repl>_r.*)?"' \
  test_python_and_bash_find_the_same_summaries

mutate "identity.py: the head-seed split stops splitting" "$IDENTITY_PY" \
'    m = HEAD_SEED_RE.search(slug)'"$SEP"'    m = None' \
  test_python_parses_back_what_it_builds test_strip_tags

# --- eval_arm.sh: the cell the number is filed under ---------------------
mutate "eval: cell name drops the replicate" "$EVAL" \
'CELL="$(eval_cell_name "${ARM}${CELL_TAG}" "$BB_STEP_K" "$REPL_TAG" "$HEAD_STEPS" "$HEAD_SEED")"'"$SEP"'CELL="$(eval_cell_name "${ARM}${CELL_TAG}" "$BB_STEP_K" "" "$HEAD_STEPS" "$HEAD_SEED")"' \
  test_390_eval_arm_replicate_gets_its_own_cell \
  test_390_eval_arm_replicate_does_not_lift_the_base_runs_number

mutate "eval: cell name drops the head seed" "$EVAL" \
'CELL="$(eval_cell_name "${ARM}${CELL_TAG}" "$BB_STEP_K" "$REPL_TAG" "$HEAD_STEPS" "$HEAD_SEED")"'"$SEP"'CELL="$(eval_cell_name "${ARM}${CELL_TAG}" "$BB_STEP_K" "$REPL_TAG" "$HEAD_STEPS" "$EVAL_DEFAULT_HEAD_SEED")"' \
  test_390_eval_arm_head_seed_gets_its_own_cell \
  test_390_eval_arm_head_seed_does_not_lift_the_other_seeds_number \
  test_390_eval_arm_head_seed_trains_its_own_head

mutate "eval: the head seed default is restated here" "$EVAL" \
'HEAD_SEED="${HEAD_SEED:-$EVAL_DEFAULT_HEAD_SEED}"'"$SEP"'HEAD_SEED="${HEAD_SEED:-20260722}"' \
  test_head_seed_default_is_379s_literal

mutate "eval: head name drops the replicate" "$EVAL" \
'HEAD_NAME="qhead_2L_${NAME}_bb${BB_STEP_K}k${REPL_TAG}"'"$SEP"'HEAD_NAME="qhead_2L_${NAME}_bb${BB_STEP_K}k"' \
  test_390_eval_arm_replicate_trains_its_own_head

mutate "eval: base cell renamed by a non-empty token" "$EVAL" \
'REPL_TAG="$(replicate_tag "$NAME" "$BB_STEP_K" "$BB")" || {'"$SEP"'REPL_TAG="_r$(basename "$BB" | md5sum | cut -c1-2)"; false || {' \
  test_390_eval_arm_base_cell_keeps_its_committed_name

mutate "eval: summary drops the backbone line" "$EVAL" \
'{ echo "$AGG"; echo "backbone: $BB"; } > "$OUT/summary.txt"'"$SEP"'echo "$AGG" > "$OUT/summary.txt"' \
  test_390_eval_arm_writes_the_backbone_into_both_summaries

mutate "eval: the aggregate stops being the first line" "$EVAL" \
'{ echo "$AGG"; echo "backbone: $BB"; } > "$OUT/summary.txt"'"$SEP"'{ echo "backbone: $BB"; echo "$AGG"; } > "$OUT/summary.txt"' \
  test_390_eval_arm_summary_still_opens_with_the_aggregate

# --- eval_arm.sh: one exit code, one operator action ---------------------
mutate "eval: partial eval reuses the ambiguity code" "$EVAL" \
'E_SETUP=20; E_NO_HEAD=21; E_PARTIAL=22; E_NO_AGGREGATE=23'"$SEP"'E_SETUP=2; E_NO_HEAD=4; E_PARTIAL=5; E_NO_AGGREGATE=6' \
  test_390_eval_arm_partial_eval_is_not_the_ambiguity_code \
  test_390_eval_arm_missing_aggregate_is_not_the_override_code \
  test_390_eval_arm_setup_failure_is_not_a_resolver_code

mutate "eval: exit codes undocumented" "$EVAL" \
"# Exit codes. 2-6 are \`resolve_eval_checkpoint.sh\`'s, propagated verbatim${SEP}# Notes." \
  test_390_eval_arm_documents_its_own_exit_codes

# --- eval_2L_gm_mase.sh: #379's call site, same two defects --------------
mutate "379 eval: cell name drops the replicate" "$EVAL379" \
'CELL="$(eval_cell_name "$ARM" "$BB_STEP_K" "$REPL_TAG" "$HEAD_STEPS" "$HEAD_SEED")"'"$SEP"'CELL="$(eval_cell_name "$ARM" "$BB_STEP_K" "" "$HEAD_STEPS" "$HEAD_SEED")"' \
  test_379_eval_names_its_cell_after_the_replicate

mutate "379 eval: the bad-tag abort reuses the setup code" "$EVAL379" \
'  exit $E_BAD_TAG; }'"$SEP"'  exit 20; }' \
  test_the_bad_tag_code_is_the_librarys_and_both_evals_use_it

mutate "379 eval: summary drops the backbone line" "$EVAL379" \
'  { echo "$AGG"; echo "backbone: $BB"; } > "$OUT/summary.txt"'"$SEP"'  echo "$AGG" > "$OUT/summary.txt"' \
  test_379_eval_writes_the_backbone_into_its_summary

mutate "379 eval: script path resolved logically" "$EVAL379" \
'ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"'"$SEP"'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"' \
  test_eval_script_takes_the_resolver_from_its_own_checkout \
  test_379_eval_finds_its_own_resolver_through_a_symlink

# --- the readers: cells looked up by name rather than by coordinate ------
# Each of these is the same defect in a different file: a replicate-backed
# cell reads as missing, and the number quietly leaves the table, the figure
# or the batch's skip check.
mutate "gate: replicate cells read as missing" "$GATE" \
'    return [str(p) for p in cid.cell_paths(root, arm, bb_k, head_steps,
                                           suffix="_summary.txt")]'"$SEP"'    import os, re
    pattern = re.compile(
        rf"^{re.escape(arm)}_bb{bb_k}k_hd{head_steps}s_summary\.txt$")
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(os.path.join(root, n) for n in names if pattern.match(n))' \
  test_the_wave3_gate_finds_a_replicate_backed_cell

mutate "bootstrap: replicate cells read as missing" "$BOOT" \
'    hits = [d / "all_results.csv"
            for d in cid.cell_paths(os.path.join(results, "eval_gm_mase"),
                                    arm, bb_k, hd)
            if (d / "all_results.csv").is_file()]'"$SEP"'    from pathlib import Path as _P
    _p = _P(results) / "eval_gm_mase" / f"{arm}_bb{bb_k}k_hd{hd}s" / "all_results.csv"
    hits = [_p] if _p.is_file() else []' \
  test_the_bootstrap_finds_a_replicate_backed_cell

mutate "bootstrap: two replicate cells resolved by picking one" "$BOOT" \
'    if len(hits) > 1:'"$SEP"'    if False:' \
  test_the_bootstrap_refuses_two_replicates_of_one_cell

mutate "figures: replicate cells read as missing" "$CELLS" \
'    hits = cid.cell_paths(root, slug, bb, hd, suffix=suffix)'"$SEP"'    hits = [p for p in [root / f"{slug}_bb{bb}k_hd{hd}s{suffix}"] if p.exists()]' \
  test_the_figures_find_a_replicate_backed_cell

mutate "figures: two replicate cells resolved by picking one" "$CELLS" \
'    if len(hits) > 1:'"$SEP"'    if False:' \
  test_the_figures_refuse_two_replicates_of_one_cell

mutate "student batch: replicate cells read as missing" "$BATCH" \
'  mapfile -t done_cells < <(eval_cell_summaries "$EVAL_ROOT" \
    "${arm}_alignstudent" "$BB_STEP_K" "$HEAD_STEPS" "$EVAL_DEFAULT_HEAD_SEED")
  if [ "${#done_cells[@]}" -gt 0 ]; then'"$SEP"'  done_cells=("$EVAL_ROOT/${arm}_alignstudent_bb${BB_STEP_K}k_hd${HEAD_STEPS}s_summary.txt")
  if [ -f "${done_cells[0]}" ]; then' \
  test_the_student_batch_skips_a_replicate_backed_cell \
  test_the_student_batch_does_not_skip_an_unmeasured_cell

# --- make_gm_table.py: two replicates are two measurements ---------------
mutate "table: the pairing throws the replicate away" "$TABLE" \
'    return row["arm_slug"], row["bb_steps"], bb_replicate(row)'"$SEP"'    return row["arm_slug"], row["bb_steps"], ""' \
  test_gm_table_pairing_keys_on_the_replicate

mutate "table: the two replicates collapse into one row" "$TABLE" \
'        rows.append({'"$SEP"'        rows = [r for r in rows if r["arm_slug"] != arm]
        rows.append({' \
  test_gm_table_keeps_a_row_per_replicate

mutate "gate: two replicate cells resolved by picking one" "$GATE" \
'    if len(paths) > 1:
        return None, AMBIGUOUS'"$SEP"'    if False:
        return None, AMBIGUOUS' \
  test_the_wave3_gate_refuses_two_replicate_cells

restore
echo
if [ "$fails" -eq 0 ]; then
  echo "ALL MUTATIONS CAUGHT"
else
  echo "$fails MUTATION(S) SURVIVED"
fi
exit "$fails"
