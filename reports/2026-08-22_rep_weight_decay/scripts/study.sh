#!/bin/bash
# #409 — one configuration, four L_rep weight floors. Sourced, never run.
#
# The configuration is #373's cell `arm6_v2_combab_alignS` at k = 3 under the
# default `sum` reduction: `cosine_similarity_batch_rep_only`, L_align on the
# student latent at weight 1.0, MoCo rep keys, tau_rep = 1.0, no CPC
# auxiliary, SIGReg on the embedding and the encoding at weight 1.0,
# tau = 0.10, EMA momentum 0.9 rising to 1.0 at step 100,000. That cell holds
# the project's best GM-Relative MASE, 1.0660 at 200,000 steps and 1.0862 at
# the 40,000-step stop this card measures.
#
# This card changes ONE hyperparameter: the weight on L_rep, which for this
# loss shape is the weight on the whole main loss. Eight arms, in
# scripts/arms.tsv. Everything else is fixed.
#
# ---- Why the weight ----------------------------------------------------------
#
# L_rep holds 92 to 93 percent of the total loss at step 40,000 and reaches its
# level near step 100. The term that moves after step 500 is L_align. So most
# of the objective is a term that stopped moving, and the card asks what
# happens when it is decayed out.
#
# ---- What this study does not write ------------------------------------------
#
# The card compares its arms to published #373 numbers. A path shared with
# #373 or #404 overwrites one of them, so four names carry the arm: the
# checkpoint root, the run name, the tag and the score file.

CF409_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF409_STUDY="$(dirname "$CF409_SCRIPTS")"
CF409_REPO="$(cd "$CF409_STUDY/../.." && pwd)"
# #373's directory, which holds the runner, the head script and the per-domain
# splitter this study reuses. Resolved from this file, so a checkout at any
# path works.
CF409_PARENT="$(cd "$CF409_STUDY/../2026-08-08_rollout_depth" && pwd)"
# The checkout the trainer and the head trainer come from. `run_leg_k.sh`
# reads `$WT/experiments/...`, so it must be a checkout, not a results dir.
CF409_WT="${WT:-$CF409_REPO}"

# ---- The configuration, which every arm shares -------------------------------
CF409_CELL="arm6_v2_combab_alignS"
CF409_K=3
CF409_REDUCE="sum"
# One stop. The card trains each arm to 40,000 backbone steps, because that is
# where the published k = 3 number sits.
CF409_STOPS="40000"
CF409_HEAD_STEPS=30000
CF409_HEAD_SEED=20260722
CF409_ENC="student"
CF409_SEED_DEFAULT="${CF409_SEED_DEFAULT:-20260520}"
CF409_ALIGN_TARGET_DEFAULT="${CF409_ALIGN_TARGET_DEFAULT:-student}"
# The weight on L_rep at step 0. Every arm starts here, and the arms table
# says where each one ends.
CF409_REP_W_START="${CF409_REP_W_START:-1.0}"
CF409_ARMS_TSV="${CF409_ARMS_TSV:-$CF409_SCRIPTS/arms.tsv}"

# ---- Trial mode --------------------------------------------------------------
# `CF409_TRIAL=<backbone steps>` runs the WHOLE pipeline at a budget that
# finishes in minutes: the same wrappers, the same #373 runner, the same head
# script and the same guards. One arm to 40,000 steps takes hours, so a trial
# puts the first wiring defect minutes from the start.
#
#   CF409_TRIAL=400 bash scripts/run_arm.sh dec0_s20 400
#
# The ramp scales with the budget, so a trial still crosses its whole decay.
if [ -n "${CF409_TRIAL:-}" ]; then
  CF409_STOPS="$CF409_TRIAL"
  CF409_HEAD_STEPS=$(( CF409_TRIAL / 2 ))
  CF409_RAMP_SCALE="${CF409_RAMP_SCALE:-$(( CF409_TRIAL * 10000 / 40000 ))}"
fi

# ---- Where the artefacts live ------------------------------------------------
#
# The durable root. Never /tmp, never inside the checkout (CLAUDE.md
# checkpoint safety rule 4), and never #373's or #404's root — one root for
# two studies is a sync loop that cannot tell their checkpoints apart.
CF409_ROOT_DEFAULT="/home/jupyter/checkpoints_backup/cf-409"
CF409_ROOT="${CF409_ROOT:-$CF409_ROOT_DEFAULT}"
CF409_RESULTS="${CF409_RESULTS:-$CF409_STUDY/results}"
CF409_PLOTS="${CF409_PLOTS:-$CF409_STUDY/plots}"

# A trial writes nowhere the study writes, and the suffix is applied once: a
# launcher exports the suffixed value and its children source this file again.
if [ -n "${CF409_TRIAL:-}" ]; then
  case "${CF409_ROOT%/}" in
    *-trial) CF409_ROOT="${CF409_ROOT%/}" ;;
    *) CF409_ROOT="${CF409_ROOT%/}-trial" ;;
  esac
  case "${CF409_RESULTS%/}" in
    */trial) CF409_RESULTS="${CF409_RESULTS%/}" ;;
    *) CF409_RESULTS="${CF409_RESULTS%/}/trial" ;;
  esac
fi

# ---- The arms ----------------------------------------------------------------

# The arms table is the one place this study's arms live, so a missing file is
# a study with no arms — and every guard below would then refuse every arm
# with a message about the arm rather than about the file.
[ -f "$CF409_ARMS_TSV" ] || {
  echo "ABORT: no arms table at $CF409_ARMS_TSV" >&2
  return 2 2>/dev/null || exit 2; }

# Every arm name, in the card's order.
cf409_arms(){
  awk -F'\t' '!/^#/ && NF >= 4 { print $1 }' "$CF409_ARMS_TSV"
}
CF409_ARMS="$(cf409_arms | tr '\n' ' ')"
CF409_ARMS="${CF409_ARMS% }"

# One arm's row, as `<arm> <end> <ramp> <seed> <align_target>`. Prints
# nothing, and returns non-zero, for an arm the table does not hold.
cf409_arm_row(){  # <arm>
  awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $1, $2, $3, $4, $5; found = 1 }
     END { exit !found }' "$CF409_ARMS_TSV"
}

cf409_require_arm(){  # <arm>
  cf409_arm_row "${1:?arm}" >/dev/null && return 0
  echo "ABORT: '$1' is not an arm of #409. The arms are: $CF409_ARMS" >&2
  return 2
}

cf409_require_stop(){  # <stop steps>
  local s
  for s in $CF409_STOPS; do [ "$s" = "${1:?stop}" ] && return 0; done
  echo "ABORT: '$1' is not a stop of #409. The stops are: $CF409_STOPS" >&2
  return 2
}

# The weight this arm's L_rep ends at, or `-` for an arm that holds it.
cf409_rep_end(){  # <arm>
  cf409_arm_row "${1:?arm}" | awk '{print $2}'
}

# How long the decay is, in steps. `0` for an arm that holds the weight.
#
# A trial scales the ramp with the budget through CF409_RAMP_SCALE, so a
# 400-step trial still crosses its whole decay and its `rep_w` column still
# reaches the floor.
cf409_ramp(){  # <arm>
  local v
  v="$(cf409_arm_row "${1:?arm}" | awk '{print $3}')" || return 1
  case "$v" in ''|-) printf '0\n'; return 0 ;; esac
  if [ -n "${CF409_RAMP_SCALE:-}" ]; then printf '%s\n' "$CF409_RAMP_SCALE"
  else printf '%s\n' "$v"; fi
}

# The backbone seed of one arm, from column 4. Three arms of this card are a
# REPEAT of another at a second seed, and those pairs are the only thing that
# measures this cell's own run-to-run spread.
cf409_seed(){  # <arm>
  local v
  v="$(cf409_arm_row "${1:?arm}" | awk '{print $4}')" || return 1
  case "$v" in ''|-) printf '%s\n' "$CF409_SEED_DEFAULT" ;;
               *) printf '%s\n' "$v" ;; esac
}

# Whose h_{t+1} L_align pulls toward, from column 5.
#
# Seven arms take the cell's own `student`. `dec0T_s20` takes `teacher`: past
# its ramp that arm holds no L_rep, so L_align against a detached EMA target
# is exactly BYOL. It is the arm most likely to hold the contrastive AUC at
# weight 0.0.
cf409_align_target(){  # <arm>
  local v
  v="$(cf409_arm_row "${1:?arm}" | awk '{print $5}')" || return 1
  case "$v" in ''|-) printf '%s\n' "$CF409_ALIGN_TARGET_DEFAULT" ;;
               *) printf '%s\n' "$v" ;; esac
}

# The trainer flags of one arm's decay, as ONE unit.
#
# An arm that holds the weight passes NEITHER flag: train.py reads "no end
# value" as "the weight is constant", and there is no value of the flag that
# means the same. So the control arms carry the cell's command line unchanged,
# which is what makes them a control.
cf409_decay_args(){  # <arm>
  local end ramp
  end="$(cf409_rep_end "${1:?arm}")" || return 1
  case "$end" in ''|-) printf -- '--rep-loss-weight %s\n' "$CF409_REP_W_START"
                      return 0 ;; esac
  ramp="$(cf409_ramp "$1")"
  printf -- '--rep-loss-weight %s --rep-loss-weight-end %s --rep-loss-weight-ramp-steps %s\n' \
    "$CF409_REP_W_START" "$end" "$ramp"
}

# The same three values as the trainer's own command line reports them, so a
# leg's log can be read against the arms table. `-` for a flag the command
# line does not carry.
cf409_decay_sig(){  # <arm>
  local end
  end="$(cf409_rep_end "${1:?arm}")" || return 1
  case "$end" in ''|-) printf '%s - -\n' "$CF409_REP_W_START"; return 0 ;; esac
  printf '%s %s %s\n' "$CF409_REP_W_START" "$end" "$(cf409_ramp "$1")"
}

# The same three values, read off a trainer command line on stdin.
cf409_decay_of_cmdline(){
  awk '{
    w = "-"; e = "-"; r = "-"
    for (i = 1; i <= NF; i++) {
      if ($i == "--rep-loss-weight") w = $(i + 1)
      if ($i == "--rep-loss-weight-end") e = $(i + 1)
      if ($i == "--rep-loss-weight-ramp-steps") r = $(i + 1)
    }
    print w, e, r }'
}

cf409_seed_of_cmdline(){
  awk '{ for (i = 1; i <= NF; i++) if ($i == "--seed") print $(i + 1) }'
}

cf409_align_target_of_cmdline(){
  awk '{ t = "-"
    for (i = 1; i <= NF; i++) if ($i == "--align-target") t = $(i + 1)
    print t }'
}

# The weight an arm HOLDS at a given step, which is not the weight its command
# line names. This is the number that compares two arms at one point of
# training.
#
# The formula is `src.models.linear_schedule_at_step`, which is linear and
# clamps the step into the ramp. It is repeated here, and not imported,
# because the shell readers of this study must not need a Python interpreter
# to print a table. `scripts/test_rep_w_at.sh` holds the two against each
# other.
cf409_rep_w_at(){  # <arm> <step>
  local end ramp
  end="$(cf409_rep_end "${1:?arm}")" || return 1
  case "$end" in ''|-) printf '%.3f\n' "$CF409_REP_W_START"; return 0 ;; esac
  ramp="$(cf409_ramp "$1")"
  awk -v w="$CF409_REP_W_START" -v e="$end" -v r="$ramp" -v s="${2:?step}" 'BEGIN{
    if (r + 0 <= 0) { printf "%.3f\n", e; exit }
    f = s / r; if (f > 1) f = 1; if (f < 0) f = 0;
    printf "%.3f\n", w + f * (e - w) }'
}

# ---- Names and paths ---------------------------------------------------------

# The run name `run_leg_k.sh` builds for this cell, through its RUN_SUFFIX. It
# carries the study and the arm, so no checkpoint of this study can be read as
# #373's, as #404's, or as another arm's.
cf409_run_suffix(){  # <arm>
  printf '_cf409_%s\n' "${1:?arm}"
}

cf409_run_name(){  # <arm>
  printf 'cf393_%s_cf373k%s%s\n' "$CF409_CELL" "$CF409_K" \
    "$(cf409_run_suffix "${1:?arm}")"
}

# The root ONE arm saves under. The arms are one cell, so `run_leg_k.sh` would
# lay all eight into one <root>/<cell>/leg_40k/. The run names differ, but a
# save dir shared by eight runs is CLAUDE.md checkpoint safety rule 3.
cf409_arm_root(){  # <arm>
  printf '%s/%s\n' "$CF409_ROOT" "${1:?arm}"
}

# Where a leg saves. `run_leg_k.sh` lays out <root>/<cell>/leg_<N>k/.
cf409_leg_dir(){  # <arm> <stop steps>
  printf '%s/%s/leg_%dk\n' "$(cf409_arm_root "${1:?arm}")" "$CF409_CELL" \
    "$(( ${2:?stop} / 1000 ))"
}

# The log #373's runner writes a leg's trainer output to.
cf409_leg_log(){  # <arm>
  printf '%s/run_%s.log\n' "$CF409_RESULTS" "$(cf409_run_name "${1:?arm}")"
}

# The losses CSV of one arm. It carries the `auc`, `rep_w`, `l_rep` and
# `l_align` columns the card reads.
cf409_losses_csv(){  # <arm> <stop steps>
  printf '%s/%s_losses.csv\n' "$(cf409_leg_dir "${1:?arm}" "${2:?stop}")" \
    "$(cf409_run_name "$1")"
}

# How many trainer command lines a leg log holds. The count before a leg
# starts is what tells this leg's line from the lines of the legs below it —
# the runner appends to one log per cell.
cf409_cmdlines(){  # <log>
  grep -c '^Command line:' "${1:?log}" 2>/dev/null || printf '0\n'
}

cf409_last_cmdline(){  # <log>
  grep '^Command line:' "${1:?log}" 2>/dev/null | tail -1
}

# Stop a runner and everything below it. A `kill` on the runner alone leaves
# the trainer holding the GPU.
cf409_kill_tree(){  # <pid>
  local pid="${1:?pid}"
  pkill -P "$pid" 2>/dev/null
  kill "$pid" 2>/dev/null
  sleep 2
  pkill -9 -P "$pid" 2>/dev/null
  kill -9 "$pid" 2>/dev/null
}
