#!/bin/bash
# What an evaluation cell is called, and what decides it.
#
#   source "$(dirname "$0")/../../../scripts/eval_cell_identity.sh"
#
# Two things decide a cell's number and so belong in its name: the backbone
# replicate, and the head seed.
#
# A resumed run is renamed `<name>_r<N>` by train.py's `safe_run_name`, so one
# (run name, step) pair can leave several backbones on disk. They are
# different models. `resolve_eval_checkpoint.sh` refuses to choose between
# them; these functions make the *output* say which one it used.
#
# The head seed is the head trainer's `--seed` — its init and its data order.
# A cell measured under another seed is another measurement.
#
# Without either token in the name, the choice is made and then thrown away:
# the cell directory, the head checkpoint and the aggregate are named from
# (arm, step, head steps) alone, so a second replicate or a second seed lands
# in the first one's directory, head-train skips on the first one's head and
# the 97-row check lifts the first one's aggregate. The run then reports a
# number it never measured, and exits 0.
#
# Each token is a function of its own input and nothing else — not of what
# happens to be on disk — so `_r3` maps to one cell whatever is added or
# deleted around it. The base run and the wave's own head seed have empty
# tokens, so the names the report cites do not move.
#
# `eval_cell_identity.py` is the same grammar for the Python readers.
# tests/test_eval_cell_identity.py generates every name both ways and
# requires them to agree.

# The head seed every wave, and so every committed cell, was measured under —
# #379's literal. Moving it renames every cell.
EVAL_DEFAULT_HEAD_SEED=20260722

# `replicate_tag` refusing is one operator action — name the replicate you
# mean — so it is one exit code in every caller.
E_BAD_TAG=25

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

# The head-seed token: empty for the seed every wave ran, `_s<seed>` for any
# other. Empty for the default for the same reason the base run's replicate
# token is — the committed cell names must not move.
head_seed_tag() {  # <head-seed>
  [ "$1" = "$EVAL_DEFAULT_HEAD_SEED" ] || printf '_s%s' "$1"
}

# The output cell name. Each token sits beside the thing it qualifies: the
# seed beside the slug whose head it trained, the replicate beside the
# backbone step it names. `arm5_nse_bb200k_r3_hd30000s` is arm5_nse's r3
# backbone at 200k under a 30 000-step head at the wave seed;
# `arm5_nse_s20260723_bb200k_hd30000s` is the same backbone under another.
#
# The head seed is not optional. Callers run under `set -u`, so one that
# leaves it out fails loudly instead of quietly naming the default cell —
# which is the difference between an identity and a convention. `$5` is bound
# to a local FIRST, in the caller's own shell: read inside the `$( )` below
# it would kill only the substitution subshell, which prints the error and
# then hands back an empty token, i.e. the default cell name, exit 0.
eval_cell_name() {  # <slug> <step-k> <replicate-tag> <head-steps> <head-seed>
  local seed="$5"
  printf '%s%s_bb%sk%s_hd%ss' "$1" "$(head_seed_tag "$seed")" "$2" "$3" "$4"
}

# Every cell summary for one (slug, backbone step, head steps, head seed)
# coordinate, whichever replicate wrote it, one path per line. Callers that
# look a cell up by name need this: the wave-2 and wave-3 backbones of this
# experiment are all resumes, so a lookup for the untagged name alone reads a
# measured cell as missing. More than one line is the caller's ambiguity to
# refuse, the same one the resolver refuses.
eval_cell_summaries() {  # <eval-root> <slug> <step-k> <head-steps> <head-seed>
  local seed="$5" slug f
  slug="$2$(head_seed_tag "$seed")"
  for f in "$1/$(eval_cell_name "$2" "$3" "" "$4" "$seed")_summary.txt" \
           "$1/${slug}_bb$3k"_r[0-9]*"_hd$4s_summary.txt"; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done
  return 0
}
