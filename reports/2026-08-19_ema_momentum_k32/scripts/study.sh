#!/bin/bash
# #404 — one configuration, four EMA momenta. Sourced, never run.
#
# The configuration is #373's cell `arm6_v2_combab_alignT`: `L_align` against
# the EMA TEACHER latent, `cosine_similarity_batch_rep_only`, MoCo rep keys,
# tau_rep = 1.0, no CPC auxiliary, SIGReg on the embedding and the encoding at
# weight 1.0, tau = 0.10. #401's k = 32 arm is that cell's student twin at the
# same depth and the same reduction, so the two numbers compare.
#
# This card changes ONE hyperparameter, the EMA momentum alpha. Four arms, in
# scripts/arms.tsv. Everything else — the depth k = 32, the mean over the
# depth copies, the backbone shape, the seed, the dataset — is fixed.
#
# ---- Why alpha, and why the teacher ------------------------------------------
#
# The teacher IS the EMA. Alpha sets how fast its weights follow the student's,
# so alpha sets the target `L_align` is trained against. #373 measured the
# schedule worth 5.5% on the teacher cell and 1.8% on the student cell. #401
# moved the ramp from 100k to 30k at k = 32 and moved the score 2.5%. Neither
# card tuned alpha AT k = 32.
#
# ---- What this study does not write ------------------------------------------
#
# The card compares its four arms to five published numbers (scripts/
# references.py). A path shared with #373 or #401 overwrites one of them, so
# four names carry the arm: the checkpoint root, the run name, the tag and the
# score file. `cf404_run_name` also carries the depth and the reduction.

CF404_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_STUDY="$(dirname "$CF404_SCRIPTS")"
CF404_REPO="$(cd "$CF404_STUDY/../.." && pwd)"
# #373's directory, which holds the runner, the head script and the per-domain
# splitter this study reuses. Resolved from this file, so a checkout at any
# path works.
CF404_PARENT="$(cd "$CF404_STUDY/../2026-08-08_rollout_depth" && pwd)"
# The checkout the trainer and the head trainer come from. `run_leg_k.sh`
# reads `$WT/experiments/...`, so it must be a checkout, not a results dir.
CF404_WT="${WT:-$CF404_REPO}"

# ---- The configuration, which every arm shares -------------------------------
CF404_CELL="arm6_v2_combab_alignT"
CF404_K=32
CF404_REDUCE="mean"
# One stop. The card trains each arm to 40,000 backbone steps and compares at
# bb40k, because that is where the published k = 3 and k = 32 numbers sit.
CF404_STOPS="40000"
CF404_HEAD_STEPS=30000
CF404_ENC="student"
CF404_ARMS_TSV="${CF404_ARMS_TSV:-$CF404_SCRIPTS/arms.tsv}"

# ---- Trial mode --------------------------------------------------------------
# `CF404_TRIAL=<backbone steps>` runs the WHOLE pipeline at a budget that
# finishes in minutes: the same wrappers, the same #373 runner, the same head
# script and the same guards over the replaced lists. One arm to 40,000 steps
# is about 5.4 hours, so a trial puts the first wiring defect minutes from the
# start instead of hours.
#
#   CF404_TRIAL=400 bash scripts/run_arm.sh a08 400
if [ -n "${CF404_TRIAL:-}" ]; then
  CF404_STOPS="$CF404_TRIAL"
  CF404_HEAD_STEPS=$(( CF404_TRIAL / 2 ))
fi

# ---- Where the artefacts live ------------------------------------------------
#
# The durable root. Never /tmp, never inside the checkout (CLAUDE.md
# checkpoint safety rule 4), and never #373's or #401's root — one root for
# two studies is a sync loop that cannot tell their checkpoints apart.
CF404_ROOT_DEFAULT="/home/jupyter/checkpoints_backup/cf-404"
CF404_ROOT="${CF404_ROOT:-$CF404_ROOT_DEFAULT}"
CF404_RESULTS="${CF404_RESULTS:-$CF404_STUDY/results}"
CF404_PLOTS="${CF404_PLOTS:-$CF404_STUDY/plots}"

# A trial writes nowhere the study writes, and the suffix is applied once: a
# launcher exports the suffixed value and its children source this file again.
if [ -n "${CF404_TRIAL:-}" ]; then
  case "${CF404_ROOT%/}" in
    *-trial) CF404_ROOT="${CF404_ROOT%/}" ;;
    *) CF404_ROOT="${CF404_ROOT%/}-trial" ;;
  esac
  case "${CF404_RESULTS%/}" in
    */trial) CF404_RESULTS="${CF404_RESULTS%/}" ;;
    *) CF404_RESULTS="${CF404_RESULTS%/}/trial" ;;
  esac
fi

# ---- Which machine owns what -------------------------------------------------
#
# The BOX trains the four backbones, two GPUs, two arms at a time. elisa trains
# every head and runs every 97-config GIFT-Eval, because the eval data and the
# `gift_eval` package live there.
#
# The two machines read two roots and the two are ONE tree: the box saves to
# CF404_BOX_RUNS on its own disk, the sync loop pulls that tree into
# CF404_SYNC_DIR keeping the relative paths, and CF404_SYNC_ROOT is what elisa
# reads. Each launcher takes its own role's root through `cf404_use_root`.
CF404_BOX_LABEL="${CF404_BOX_LABEL:-box_a}"
CF404_BOX_RUNS="${CF404_BOX_RUNS:-/root/cf404_runs}"
CF404_SYNC_DIR="${CF404_SYNC_DIR:-$HOME/cf404_sync/$CF404_BOX_LABEL}"
CF404_SYNC_ROOT="${CF404_SYNC_ROOT:-$CF404_SYNC_DIR/sync}"

# ---- The arms ----------------------------------------------------------------

# Every arm name, in the card's order.
cf404_arms(){
  awk -F'\t' '!/^#/ && NF >= 4 { print $1 }' "$CF404_ARMS_TSV"
}
CF404_ARMS="$(cf404_arms | tr '\n' ' ')"
CF404_ARMS="${CF404_ARMS% }"

# One arm's row, as `<arm> <tau> <end> <ramp>`. Prints nothing, and returns
# non-zero, for an arm the table does not hold.
cf404_arm_row(){  # <arm>
  awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $1, $2, $3, $4; found = 1 }
     END { exit !found }' "$CF404_ARMS_TSV"
}

# The arm's alpha at step 0, which is the x axis of the card's first figure.
cf404_alpha(){  # <arm>
  cf404_arm_row "${1:?arm}" | awk '{print $2}'
}

# `fixed` or `ramp`, which is the marker of that figure.
cf404_schedule(){  # <arm>
  local end
  end="$(cf404_arm_row "${1:?arm}" | awk '{print $3}')" || return 1
  case "$end" in
    ""|-) printf 'fixed\n' ;;
    *) printf 'ramp\n' ;;
  esac
}

# The trainer flags of one arm's EMA momentum, as ONE unit.
#
# A fixed arm passes `--ema-tau` alone. It does NOT pass `--ema-tau-end` at
# any value: train.py reads "no end value" as "alpha constant", and there is
# no value of the flag that means the same. This is why the flags REPLACE
# #373's schedule (`EMA_ARGS`) rather than appending to it — a repeat can
# change a flag, never remove one.
cf404_ema_args(){  # <arm>
  local row tau end ramp
  row="$(cf404_arm_row "${1:?arm}")" || return 1
  read -r _ tau end ramp <<<"$row"
  if [ "$end" = "-" ]; then
    printf -- '--ema-tau %s\n' "$tau"
  else
    printf -- '--ema-tau %s --ema-tau-end %s --ema-tau-ramp-steps %s\n' \
      "$tau" "$end" "$ramp"
  fi
}

# ---- Names and paths ---------------------------------------------------------

# The run name `run_leg_k.sh` builds for this cell, through its RUN_SUFFIX. It
# carries the reduction and the arm, so no checkpoint of this study can be
# read as #373's, as #401's, or as another arm's.
cf404_run_suffix(){  # <arm>
  printf '_%s_%s\n' "$CF404_REDUCE" "${1:?arm}"
}

cf404_run_name(){  # <arm>
  printf 'cf393_%s_cf373k%s%s\n' "$CF404_CELL" "$CF404_K" \
    "$(cf404_run_suffix "${1:?arm}")"
}

# The root ONE arm saves under. The four arms are one cell, so `run_leg_k.sh`
# would lay all four into one <root>/<cell>/leg_40k/. The run names differ, but
# a save dir shared by four runs is CLAUDE.md checkpoint safety rule 3.
cf404_arm_root(){  # <arm>
  printf '%s/%s\n' "$CF404_ROOT" "${1:?arm}"
}

# Where a leg saves. `run_leg_k.sh` lays out <root>/<cell>/leg_<N>k/.
cf404_leg_dir(){  # <arm> <stop steps>
  printf '%s/%s/leg_%dk\n' "$(cf404_arm_root "${1:?arm}")" "$CF404_CELL" \
    "$(( ${2:?stop} / 1000 ))"
}

# The checkpoint a stop produced, or nothing.
#
# Two names, not one glob. `<name>_<N>k.pth` is the leg's own, and
# `<name>_r<N>_<N>k.pth` is train.py's `_rN` infix on a re-fired leg
# (`safe_run_name`). A trailing `*` would take both, and the optimizer file
# with them.
cf404_bb_ckpt(){  # <arm> <stop steps>
  local dir name kk
  dir="$(cf404_leg_dir "${1:?arm}" "${2:?stop}")"
  name="$(cf404_run_name "$1")"
  kk=$(( $2 / 1000 ))
  ls "$dir/$name"_"$kk"k.pth "$dir/$name"_r[0-9]*_"$kk"k.pth 2>/dev/null \
    | grep -v optimizer | head -1
}

# A step count, as a tag reads it. `40000` -> `40k`, `400` -> `400`. A trial
# budget is not a multiple of 1000, and rounding it to `0k` would give two
# budgets one tag.
cf404_steps_label(){  # <steps>
  local n="${1:?steps}"
  if [ $(( n % 1000 )) -eq 0 ]; then printf '%dk' $(( n / 1000 ))
  else printf '%d' "$n"; fi
}

# The inverse, for the reader in collect.sh.
cf404_steps_of(){  # <label>
  case "${1:?label}" in
    *k) printf '%d' $(( ${1%k} * 1000 )) ;;
    *)  printf '%d' "${1}" ;;
  esac
}

# The name of one (arm, stop, head budget). It names the head checkpoint, the
# eval directory and the score file.
cf404_tag(){  # <arm> <stop steps> <head steps>
  printf '%s_bb%s_h%s_%s\n' "${1:?arm}" \
    "$(cf404_steps_label "${2:?stop}")" \
    "$(cf404_steps_label "${3:?head steps}")" "$CF404_ENC"
}

# Where #373's head script lays one tag's head, its GIFT-Eval output and its
# merged 97-config CSV. One root per arm, the same rule as the backbones.
cf404_eval_dir(){  # <arm> <tag>
  printf '%s/eval/%s\n' "$(cf404_arm_root "${1:?arm}")" "${2:?tag}"
}

# The log #373's runner writes a leg's trainer output to.
cf404_leg_log(){  # <arm>
  printf '%s/run_%s.log\n' "$CF404_RESULTS" "$(cf404_run_name "${1:?arm}")"
}

# ---- What the trainer of a leg actually runs ---------------------------------
#
# The depth leaves a proof in the artefacts: a k-depth run writes k + 1
# `cos_err_dj` columns. The MOMENTUM and the REDUCTION leave none. Four arms
# that share a configuration write the same file names, the same CSV columns
# and the same log lines, so an arm that trained another arm's alpha is a
# duplicate under a name that says otherwise.
#
# The trainer's own command line is the one place that names both. train.py
# prints it as the first line of every run's log (#401), and the functions
# below read it back from there.

# The `--flag value` or `--flag=value` value on a command line, or nothing.
# Reads the command line on stdin, NUL-separated (a /proc cmdline) or
# space-separated (a log line).
cf404_arg_of_cmdline(){  # <flag>
  tr '\0 ' '\n\n' | awk -F= -v f="${1:?flag}" '
    $1 == f { if (NF > 1) { print $2 } else { getline; print } exit }'
}

# The EMA momentum a trainer command line names, in the shape of an arms.tsv
# row: `<tau> <end> <ramp>`, with `-` for a flag the line does not carry. So
# the comparison against the table is one string equality, not three.
cf404_ema_of_cmdline(){
  local line tau end ramp
  line="$(cat)"
  tau="$(printf '%s' "$line" | cf404_arg_of_cmdline --ema-tau)"
  end="$(printf '%s' "$line" | cf404_arg_of_cmdline --ema-tau-end)"
  ramp="$(printf '%s' "$line" | cf404_arg_of_cmdline --ema-tau-ramp-steps)"
  printf '%s %s %s\n' "${tau:--}" "${end:--}" "${ramp:--}"
}

# The same shape, out of the arms table.
cf404_ema_sig(){  # <arm>
  cf404_arm_row "${1:?arm}" | awk '{print $2, $3, $4}'
}

# The reduction a trainer command line names. `sum` when it carries no flag,
# because `sum` is train.py's own default.
cf404_reduce_of_cmdline(){
  local v; v="$(cf404_arg_of_cmdline --train-rollout-reduce)"
  printf '%s\n' "${v:-sum}"
}

# How many command lines a leg log holds. `run_leg_k.sh` APPENDS, so a resumed
# cell's log carries one per leg. A caller that counts before it starts a leg
# knows when THIS leg's line has landed.
# Always an integer, including for a log that does not exist yet — which is
# the normal case on a first leg. `grep -c` prints nothing and exits 2 on a
# missing file, and a caller comparing "" with `-le` aborts its wait loop in
# silence, so the momentum check would be skipped exactly when it is needed.
cf404_cmdlines(){  # <trainer log>
  local n
  n="$(grep -c '^Command line:' "${1:?log}" 2>/dev/null)" || n=0
  printf '%s\n' "${n:-0}"
}

# The LAST command line in a leg's log. Prints nothing, and returns non-zero,
# when the log holds none.
cf404_last_cmdline(){  # <trainer log>
  local line
  line="$(grep '^Command line:' "${1:?log}" 2>/dev/null | tail -1)"
  [ -n "$line" ] || return 1
  printf '%s' "${line#Command line: }"
}

# Every process under a tree, leaves first, so a signal reaches the trainer
# before the wrapper that would otherwise report its death first.
cf404_kill_tree(){  # <pid>
  local kid
  for kid in $(pgrep -P "${1:?pid}" 2>/dev/null); do cf404_kill_tree "$kid"; done
  kill -TERM "$1" 2>/dev/null
}

# How many sync loops run for ONE local root, identified by their working
# directory. elisa is shared: another study's loop is not this study's.
#
# `pgrep -f` also matches a process that merely NAMES the file on its command
# line, this check among them, so the argument list decides and not the
# pattern. `wc -l`, because `pgrep -c` prints 0 AND exits 1 on no match.
cf404_sync_loops(){  # <local dir>
  local want="${1:?local dir}" p n=0
  want="${want%/}"
  for p in $(pgrep -f 'sync_loop\.sh' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    tr '\0' '\n' <"/proc/$p/cmdline" 2>/dev/null \
      | grep -qx '.*/sync_loop\.sh' || continue
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$want" ] || continue
    n=$(( n + 1 ))
  done
  echo "$n"
}

# The root a machine's ROLE implies: CF404_BOX_RUNS on the box, which trains
# every backbone, and CF404_SYNC_ROOT on elisa, which reads that tree.
#
# Called by a launcher that captured `CF404_ROOT_GIVEN` BEFORE it sourced this
# file, so an operator's own root still wins. Exported, because the launcher's
# children source this file again.
cf404_use_root(){  # <root>
  CF404_ROOT="${CF404_ROOT_GIVEN:-${1:?root}}"
  export CF404_ROOT
}

# ---- The checkout this study needs -------------------------------------------
#
# Two things this card depends on are NOT in every checkout of this repository.
# A box bootstrapped from a stale branch would train, log nothing unusual, and
# hand back the wrong arm eleven hours later.
#
#   EMA_ARGS in #373's runner. Without it every arm trains the runner's own
#   0.9 -> 1.0 at 100k schedule, so the four arms are ONE arm, four times.
#   `run_arm.sh` also catches this at run time, off the trainer's command
#   line. This check is the cheap one, before the card is rented.
#
#   --train-rollout-reduce in the trainer (#401). Without it the arms train
#   the SUMMED objective, which is not the one #401's k = 32 number came from,
#   so the comparison the card is built on is gone.
#
# Prints what is missing and returns non-zero. Takes a checkout, so a launcher
# can check the box's copy as well as its own.
cf404_check_checkout(){  # [checkout]
  local wt="${1:-$CF404_WT}" missing=0 runner trainer
  runner="$wt/reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh"
  trainer="$wt/experiments/2026-04-27_freq-embedding/scripts/train.py"
  if ! grep -q 'EMA_ARGS_ARR' "$runner" 2>/dev/null; then
    echo "ABORT: $runner takes no EMA_ARGS." >&2
    echo "  Every arm would train the runner's own schedule, so the four" >&2
    echo "  arms would be one arm four times." >&2
    missing=1
  fi
  if ! grep -q -- '--train-rollout-reduce' "$trainer" 2>/dev/null; then
    echo "ABORT: $trainer has no --train-rollout-reduce." >&2
    echo "  The arms would train the SUMMED objective, and #401's k = 32" >&2
    echo "  number this card compares against is the MEAN one." >&2
    missing=1
  fi
  [ "$missing" -eq 0 ]
}

# ---- The guards --------------------------------------------------------------
#
# Every guard prints what it refused. A typo that trains for five hours is
# expensive, and four arms that differ in one character are easy to mistype.

cf404_is_in(){  # <value> <space separated list>
  case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

cf404_require_arm(){  # <arm>
  cf404_is_in "${1:-}" "$CF404_ARMS" && return 0
  echo "ABORT: arm='${1:-}' is not an arm of this study ($CF404_ARMS)" >&2
  echo "  The arms live in $CF404_ARMS_TSV." >&2
  return 2
}

cf404_require_stop(){  # <stop steps>
  cf404_is_in "${1:-}" "$CF404_STOPS" && return 0
  echo "ABORT: stop='${1:-}' is not a stop of this study ($CF404_STOPS)" >&2
  return 2
}

# The card defines ONE head budget. A tag written at any other budget is a tag
# collect.sh reads as this study's and the card never defined.
cf404_require_head_steps(){  # <head steps>
  [ "${1:-}" = "$CF404_HEAD_STEPS" ] && return 0
  echo "ABORT: head steps='${1:-}' is not this card's budget" \
       "($CF404_HEAD_STEPS)" >&2
  return 2
}
