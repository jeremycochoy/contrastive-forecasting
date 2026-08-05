#!/bin/bash
# The two names a backbone checkpoint decides: which replicate it is, and
# what the output cell that cites it is called.
#
#   source "$(dirname "$0")/../../../scripts/eval_cell_identity.sh"
#
# A resumed run is renamed `<name>_r<N>` by train.py's `safe_run_name`, so one
# (run name, step) pair can leave several backbones on disk. They are
# different models. `resolve_eval_checkpoint.sh` refuses to choose between
# them; these functions make the *output* say which one it used.
#
# Without that, the choice is made and then thrown away: the cell directory,
# the head checkpoint and the aggregate are named from (arm, step) alone, so
# a replicate lands in the base run's directory, head-train skips on the base
# run's head and the 97-row check lifts the base run's aggregate. The
# replicate then reports a number measured on another backbone, and exits 0.
#
# The token is a function of the filename and nothing else — not of what else
# happens to be on disk — so `_r3` maps to one cell whatever is added or
# deleted around it, and the base run's cells keep the names the report cites.

# Does <path> name a backbone snapshot of (<run-name>, <step-k>)?
# `_<step>k.pth` is anchored on both sides, so neither another step
# (`_400k.pth`) nor the optimizer sidecar (`_40k_optimizer.pth`) matches, and
# `_r[0-9]+` rather than `_r.*` keeps a sibling recipe suffix (`_revin_`) from
# reading as a resume. Leaves the match in BASH_REMATCH for `replicate_tag`.
ckpt_is_run_step() {  # <run-name> <step-k> <path>
  [[ "$(basename "$3")" =~ ^"$1"(_r[0-9]+)?_"$2"k\.pth$ ]]
}

# The replicate token of <path>: empty for the base run, `_r<N>` for a
# resume. Non-zero, and nothing on stdout, when <path> is not a snapshot of
# this (run name, step) pair — an empty token means "the base run", so a
# caller must never get one by accident.
replicate_tag() {  # <run-name> <step-k> <path>
  ckpt_is_run_step "$@" || return 1
  printf '%s' "${BASH_REMATCH[1]}"
}

# The output cell name. The replicate token sits beside the backbone step it
# qualifies: `arm5_nse_bb200k_r3_hd30000s` is arm5_nse's r3 backbone at 200k
# under a 30 000-step head.
eval_cell_name() {  # <slug> <step-k> <replicate-tag> <head-steps>
  printf '%s_bb%sk%s_hd%ss' "$1" "$2" "$3" "$4"
}

# Every cell summary for one (slug, backbone step, head steps) triple,
# whichever replicate wrote it, one path per line. Callers that look a cell
# up by name need this: the wave-2 and wave-3 backbones of this experiment
# are all resumes, so a lookup for the untagged name alone reads a measured
# cell as missing. More than one line is the caller's ambiguity to refuse,
# the same one the resolver refuses.
eval_cell_summaries() {  # <eval-root> <slug> <step-k> <head-steps>
  local f
  for f in "$1/$(eval_cell_name "$2" "$3" "" "$4")_summary.txt" \
           "$1/$2_bb$3k"_r[0-9]*"_hd$4s_summary.txt"; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done
  return 0
}
