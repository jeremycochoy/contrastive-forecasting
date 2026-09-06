#!/bin/bash
# #412 — one cell, four published configurations, at the size of a small
# foundation model. Sourced, never run.
#
# Every score of this project comes from a backbone of 720,668 trainable
# parameters. Moirai-2-Small holds 11.4 million and scores 0.728 on the same
# 97-config GIFT-Eval, where this project's best is 1.0651. So the objective
# and the size are not separated, and this card moves the size.
#
# ---- The shape, and why this one ---------------------------------------------
#
# A transformer layer of this project holds about 12 * d_model^2 parameters.
# The cell trains two stacks, 3 encoder layers and 3 forecaster layers, so 6
# layers in all. `scripts/size_table.sh` measures every candidate:
#
#   d_model 320, 3 + 3 layers    8,076,956 trainable    29% under the target
#   d_model 352, 3 + 3 layers    9,678,476 trainable    15% under
#   d_model 384, 3 + 3 layers   11,431,548 trainable     0.3% over
#   d_model 416, 3 + 3 layers   13,336,172 trainable    17% over
#
# `d_model` 384 at the published depth is the nearest, and it moves ONE axis.
# A pair at another depth reaches the same count only away from 11.4 million:
# 320 at 4 + 5 layers gives 11,769,356, which is 3.2% over, and it moves the
# width, the depth AND the balance of the two stacks at the same time. Then no
# result of this card could be read against its parents.
#
# The head count stays at 8, so the head width goes 8 to 48.
#
# ---- What the size numbers mean ----------------------------------------------
#
# The card reads 1,135,774 for the model today. That is the sum over the
# checkpoint file after the `teacher_*` keys are dropped, and it counts the
# patch encoder TWO times: that module is in the file as `encoder.*` and again
# as `transformer.input_to_latent.*`. The model trains 720,668 parameters and
# holds a frozen EMA teacher of 563,760 beside them. `src/model_size.py` gives
# the three numbers and says which one compares to a published model.
#
# ---- The trap this card must not fall into ----------------------------------
#
# `L_align` targets the EMA teacher on every arm, and the cell name says so.
# On an align-student cell the teacher reaches the loss only through the MoCo
# keys inside `L_rep` (`src/loss.py`). #409 spent a 12-hour run before it read
# that line. `run_leg_k.sh` sets `--align-target teacher` for this cell, and
# `run_arm.sh` reads it back off the trainer's own command line.
#
# ---- What this study must not write over -------------------------------------
#
# The card compares its arms against published numbers of #373, #393, #401,
# #404 and #409. Those cards trained the SAME cell, so their checkpoints, run
# names and score files collide with this card's by default. Four names carry
# this card instead: the checkpoint root, the run name, the tag and the score
# file.

CF412_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF412_STUDY="$(dirname "$CF412_SCRIPTS")"
CF412_REPO="$(cd "$CF412_STUDY/../.." && pwd)"
# #373's directory, which holds the leg runner, the head script and the
# evaluation this study reuses. Resolved from this file, so a checkout at any
# path works.
CF412_PARENT="$(cd "$CF412_STUDY/../2026-08-08_rollout_depth" && pwd)"
# The checkout the trainer comes from. `run_leg_k.sh` reads
# `$WT/experiments/...`, so it must be a checkout, not a results directory.
CF412_WT="${WT:-$CF412_REPO}"

# ---- The configuration every arm shares --------------------------------------
CF412_CELL="arm6_v2_combab_alignT"
# The shape. See the header for why 384 at 3 + 3 layers.
CF412_D_MODEL="${CF412_D_MODEL:-384}"
CF412_NUM_LAYERS="${CF412_NUM_LAYERS:-3}"
CF412_NUM_ENCODER_LAYERS="${CF412_NUM_ENCODER_LAYERS:-3}"
CF412_N_HEADS="${CF412_N_HEADS:-8}"
# The batch size is the count of negatives of this objective. It must stay at
# the published 64, or no arm compares to its parent. A wider model does not
# change that, so the value is here and `run_arm.sh` reads it back off the
# trainer's command line.
CF412_BATCH_SIZE="${CF412_BATCH_SIZE:-64}"
# The stops the parent reports use.
CF412_STOPS="${CF412_STOPS:-40000 100000 200000}"
CF412_HEAD_STEPS="${CF412_HEAD_STEPS:-30000}"
CF412_ENC="${CF412_ENC:-student}"
CF412_SEED_DEFAULT="${CF412_SEED_DEFAULT:-20260520}"
CF412_ARMS_TSV="${CF412_ARMS_TSV:-$CF412_SCRIPTS/arms.tsv}"
# Moirai-2-Small, from arXiv:2511.11698. The target the shape aims at.
CF412_TARGET_PARAMS=11400000
# A 384-wide backbone and its head take more of the card than a 64-wide one,
# so the head gate waits for more free memory than #373's default.
CF412_HEAD_VRAM_MIB="${CF412_HEAD_VRAM_MIB:-12000}"

# ---- Trial mode --------------------------------------------------------------
# `CF412_TRIAL=<backbone steps>` runs the whole pipeline at a budget that ends
# in minutes: the same wrappers, the same runner, the same head script and the
# same guards. One arm to 40,000 steps is many hours at this width, so a trial
# puts the first wiring defect minutes from the start.
#
#   CF412_TRIAL=60 bash scripts/run_arm.sh k3_r100_09 60
if [ -n "${CF412_TRIAL:-}" ]; then
  CF412_STOPS="$CF412_TRIAL"
  CF412_HEAD_STEPS=$(( CF412_TRIAL / 2 ))
fi

# ---- Where the artefacts live ------------------------------------------------
#
# Never /tmp, never inside the checkout (CLAUDE.md checkpoint safety rule 4),
# and never the root of another card.
CF412_ROOT_DEFAULT="/home/jupyter/checkpoints_backup/cf-412"
CF412_ROOT="${CF412_ROOT:-$CF412_ROOT_DEFAULT}"
CF412_RESULTS="${CF412_RESULTS:-$CF412_STUDY/results}"
CF412_PLOTS="${CF412_PLOTS:-$CF412_STUDY/plots}"

# A trial writes nowhere the study writes, and the suffix is applied one time:
# a launcher exports the suffixed value and its children source this file
# again.
if [ -n "${CF412_TRIAL:-}" ]; then
  case "${CF412_ROOT%/}" in
    *-trial) CF412_ROOT="${CF412_ROOT%/}" ;;
    *) CF412_ROOT="${CF412_ROOT%/}-trial" ;;
  esac
  case "${CF412_RESULTS%/}" in
    */trial) CF412_RESULTS="${CF412_RESULTS%/}" ;;
    *) CF412_RESULTS="${CF412_RESULTS%/}/trial" ;;
  esac
fi

# ---- The arms ----------------------------------------------------------------

[ -f "$CF412_ARMS_TSV" ] || {
  echo "ABORT: no arms table at $CF412_ARMS_TSV" >&2
  return 2 2>/dev/null || exit 2; }

# Every arm name, in the card's order.
cf412_arms(){
  awk -F'\t' '!/^#/ && NF >= 6 { print $1 }' "$CF412_ARMS_TSV"
}
CF412_ARMS="$(cf412_arms | tr '\n' ' ')"
CF412_ARMS="${CF412_ARMS% }"

# One arm's row, as `<arm> <k> <reduce> <tau> <end> <ramp> <seed>`. Prints
# nothing, and returns non-zero, for an arm the table does not hold.
cf412_arm_row(){  # <arm>
  awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $1, $2, $3, $4, $5, $6, $7; found = 1 }
     END { exit !found }' "$CF412_ARMS_TSV"
}

# The rollout depth k of one arm.
cf412_depth(){  # <arm>
  cf412_arm_row "${1:?arm}" | awk '{print $2}'
}

# The reduction over the k + 1 depth copies, `sum` or `mean`.
cf412_reduce(){  # <arm>
  cf412_arm_row "${1:?arm}" | awk '{print $3}'
}

# The backbone seed. Every arm names one, and the fifth arm is the repeat that
# measures this size's own seed band.
cf412_seed(){  # <arm>
  local v
  v="$(cf412_arm_row "${1:?arm}" | awk '{print $7}')" || return 1
  case "$v" in ''|-) printf '%s\n' "$CF412_SEED_DEFAULT" ;;
               *) printf '%s\n' "$v" ;; esac
}

# The trainer flags of one arm's EMA momentum, as ONE unit.
#
# They REPLACE `run_leg_k.sh`'s schedule rather than append to it. A repeated
# flag can change a value, never remove one, and an arm that holds alpha fixed
# must pass no `--ema-tau-end` at all. No arm of this card is fixed today, and
# the shape stays because a new row can be.
cf412_ema_args(){  # <arm>
  local row name k red tau end ramp seed
  row="$(cf412_arm_row "${1:?arm}")" || return 1
  read -r name k red tau end ramp seed <<<"$row"
  if [ "$end" = "-" ]; then
    printf -- '--ema-tau %s\n' "$tau"
  else
    printf -- '--ema-tau %s --ema-tau-end %s --ema-tau-ramp-steps %s\n' \
      "$tau" "$end" "$ramp"
  fi
}

# The same momentum, in the shape a command line reads back as: `<tau> <end>
# <ramp>`, with `-` for a flag the line does not carry. So the comparison is
# one string equality, not three.
cf412_ema_sig(){  # <arm>
  cf412_arm_row "${1:?arm}" | awk '{print $4, $5, $6}'
}

# ---- The shape, on the trainer's command line --------------------------------
#
# `run_leg_k.sh` states `--d-model 64 --num-encoder-layers 3 --num-layers 3`
# in its own shared block. These flags go LAST on the line, and argparse keeps
# the last value of a repeat. So the width this card trains is the last
# `--d-model` on the line, never the first.
cf412_arch_args(){
  printf -- '--d-model %s --n-heads %s --num-layers %s --num-encoder-layers %s\n' \
    "$CF412_D_MODEL" "$CF412_N_HEADS" "$CF412_NUM_LAYERS" \
    "$CF412_NUM_ENCODER_LAYERS"
}

# The shape as `<d_model> <num_layers> <num_encoder_layers>`, which is what
# `cf412_arch_of_cmdline` reads back off a trainer command line. Every arm
# carries one shape, so this takes no arm.
cf412_arch_sig(){
  printf '%s %s %s\n' "$CF412_D_MODEL" "$CF412_NUM_LAYERS" \
    "$CF412_NUM_ENCODER_LAYERS"
}

# The shape the head trainer and the GIFT-Eval build the backbone to. #373's
# `bb_shape.sh` reads this value, and `head_eval_bb.sh` starts `eval_local.sh`
# as a child, so one exported value reaches both.
cf412_bb_shape(){
  printf -- '--d-model %s --n-heads %s --num-layers %s\n' \
    "$CF412_D_MODEL" "$CF412_N_HEADS" "$CF412_NUM_LAYERS"
}

# ---- Names and paths ---------------------------------------------------------

# The suffix `run_leg_k.sh` puts in the run name. It carries the card and the
# arm, so no checkpoint of this study reads as #373's, #401's, #404's, #409's
# or another arm's.
cf412_run_suffix(){  # <arm>
  printf '_cf412_%s\n' "${1:?arm}"
}

cf412_run_name(){  # <arm>
  printf 'cf393_%s_cf373k%s%s\n' "$CF412_CELL" "$(cf412_depth "${1:?arm}")" \
    "$(cf412_run_suffix "$1")"
}

# The root ONE arm saves under. `run_leg_k.sh` lays every run of one cell into
# one `<root>/<cell>/leg_<N>k/`, and five runs in one save directory is
# CLAUDE.md checkpoint safety rule 3.
cf412_arm_root(){  # <arm>
  printf '%s/%s\n' "$CF412_ROOT" "${1:?arm}"
}

cf412_leg_dir(){  # <arm> <stop steps>
  printf '%s/%s/leg_%dk\n' "$(cf412_arm_root "${1:?arm}")" "$CF412_CELL" \
    "$(( ${2:?stop} / 1000 ))"
}

# The checkpoint a stop produced, or nothing. Two names, not one glob:
# `<name>_<N>k.pth` is the leg's own, and `<name>_r<N>_<N>k.pth` is train.py's
# infix on a re-fired leg. A trailing `*` would take the optimizer file too.
cf412_bb_ckpt(){  # <arm> <stop steps>
  local dir name kk
  dir="$(cf412_leg_dir "${1:?arm}" "${2:?stop}")"
  name="$(cf412_run_name "$1")"
  kk=$(( $2 / 1000 ))
  ls "$dir/$name"_"$kk"k.pth "$dir/$name"_r[0-9]*_"$kk"k.pth 2>/dev/null \
    | grep -v optimizer | head -1
}

# A step count, as a tag reads it. `40000` -> `40k`, `60` -> `60`. A trial
# budget is not a multiple of 1000, and rounding it to `0k` would give two
# budgets one tag.
cf412_steps_label(){  # <steps>
  local n="${1:?steps}"
  if [ $(( n % 1000 )) -eq 0 ]; then printf '%dk' $(( n / 1000 ))
  else printf '%d' "$n"; fi
}

# The name of one (arm, stop, head budget). It names the head checkpoint, the
# evaluation directory and the score file.
cf412_tag(){  # <arm> <stop steps> <head steps>
  printf '%s_bb%s_h%s_%s\n' "${1:?arm}" \
    "$(cf412_steps_label "${2:?stop}")" \
    "$(cf412_steps_label "${3:?head steps}")" "$CF412_ENC"
}

cf412_eval_dir(){  # <arm> <tag>
  printf '%s/eval/%s\n' "$(cf412_arm_root "${1:?arm}")" "${2:?tag}"
}

# The log #373's runner writes a leg's trainer output to.
cf412_leg_log(){  # <arm>
  printf '%s/run_%s.log\n' "$CF412_RESULTS" "$(cf412_run_name "${1:?arm}")"
}

cf412_score_file(){  # <arm> <stop steps>
  printf '%s/score_%s.txt\n' "$CF412_RESULTS" \
    "$(cf412_tag "${1:?arm}" "${2:?stop}" "$CF412_HEAD_STEPS")"
}

# Every (arm, stop) pair, one for each line.
cf412_pairs(){
  local arm stop
  for arm in $CF412_ARMS; do
    for stop in $CF412_STOPS; do printf '%s %s\n' "$arm" "$stop"; done
  done
}

# ---- What the trainer of a leg actually runs ---------------------------------
#
# The width, the depth k, the reduction, the momentum and the seed leave no
# proof in the checkpoint that a reader can see at a glance. Five arms of one
# cell write the same file names and the same CSV columns, so an arm that
# trained another arm's flags is a duplicate under a name that says otherwise.
#
# The trainer's own command line names all of them. train.py prints it as the
# first line of every run's log, and the readers below take it from there.

# The LAST value of a flag on a command line, which is the value argparse
# keeps. Reads the line on stdin, NUL-separated or space-separated.
#
# Last, not first, because `run_leg_k.sh` states the width, the depth and the
# batch size in its own block and this card repeats them at the end of the
# line. A first-hit reader would report 64 on every arm.
cf412_last_arg_of_cmdline(){  # <flag>
  tr '\0 ' '\n\n' | awk -F= -v f="${1:?flag}" '
    $1 == f { if (NF > 1) { v = $2 } else { getline; v = $0 } }
    END { if (v != "") print v }'
}

# The shape a command line names, as `<d_model> <num_layers>
# <num_encoder_layers>`, with `-` for a flag the line does not carry.
cf412_arch_of_cmdline(){
  local line d nl ne
  line="$(cat)"
  d="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --d-model)"
  nl="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --num-layers)"
  ne="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --num-encoder-layers)"
  printf '%s %s %s\n' "${d:--}" "${nl:--}" "${ne:--}"
}

# The EMA momentum a command line names, in the shape of `cf412_ema_sig`.
cf412_ema_of_cmdline(){
  local line tau end ramp
  line="$(cat)"
  tau="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --ema-tau)"
  end="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --ema-tau-end)"
  ramp="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --ema-tau-ramp-steps)"
  printf '%s %s %s\n' "${tau:--}" "${end:--}" "${ramp:--}"
}

# The rollout depth. `0` when the line carries no flag, which is train.py's
# own default.
cf412_depth_of_cmdline(){
  local v; v="$(cf412_last_arg_of_cmdline --train-rollout-depth)"
  printf '%s\n' "${v:-0}"
}

# The reduction. `sum` when the line carries no flag, which is train.py's own
# default and the value the k = 3 parent ran.
cf412_reduce_of_cmdline(){
  local v; v="$(cf412_last_arg_of_cmdline --train-rollout-reduce)"
  printf '%s\n' "${v:-sum}"
}

cf412_seed_of_cmdline(){
  local v; v="$(cf412_last_arg_of_cmdline --seed)"
  printf '%s\n' "${v:--}"
}

cf412_batch_of_cmdline(){
  local v; v="$(cf412_last_arg_of_cmdline --batch-size)"
  printf '%s\n' "${v:--}"
}

# The align target. `student` when the line carries no flag, which is the
# value this card must never train.
cf412_align_target_of_cmdline(){
  local v; v="$(cf412_last_arg_of_cmdline --align-target)"
  printf '%s\n' "${v:-student}"
}

# How many command lines a leg log holds. `run_leg_k.sh` APPENDS, so a resumed
# cell's log carries one for each leg. A caller that counts before it starts a
# leg knows when THIS leg's line has landed. Always an integer, including for
# a log that does not exist yet.
cf412_cmdlines(){  # <trainer log>
  local n
  n="$(grep -c '^Command line:' "${1:?log}" 2>/dev/null)" || n=0
  printf '%s\n' "${n:-0}"
}

# The LAST command line in a leg's log.
cf412_last_cmdline(){  # <trainer log>
  local line
  line="$(grep '^Command line:' "${1:?log}" 2>/dev/null | tail -1)"
  [ -n "$line" ] || return 1
  printf '%s' "${line#Command line: }"
}

# Stop a runner and every process below it. `kill $!` reaches the wrapper
# only, and the trainer under it keeps the card.
cf412_kill_tree(){  # <pid>
  local pid="${1:?pid}" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do cf412_kill_tree "$child"; done
  kill -TERM "$pid" 2>/dev/null
}

# ---- Guards ------------------------------------------------------------------

cf412_is_in(){  # <needle> <space separated haystack>
  local x
  for x in ${2:-}; do [ "$x" = "${1:-}" ] && return 0; done
  return 1
}

# Two numbers, compared as numbers. `3` and `3.0` are one value.
cf412_num_eq(){  # <a> <b>
  awk -v a="${1:-}" -v b="${2:-}" 'BEGIN{
    if (a == "" || b == "" || a == "-" || b == "-") exit (a != b)
    exit !(a + 0 == b + 0) }'
}

cf412_require_arm(){  # <arm>
  cf412_is_in "${1:-}" "$CF412_ARMS" && return 0
  echo "ABORT: arm='${1:-}' is not an arm of this study ($CF412_ARMS)" >&2
  echo "  The arms live in $CF412_ARMS_TSV." >&2
  return 2
}

cf412_require_stop(){  # <stop steps>
  cf412_is_in "${1:-}" "$CF412_STOPS" && return 0
  echo "ABORT: stop='${1:-}' is not a stop of this study ($CF412_STOPS)" >&2
  return 2
}

# The card defines ONE head budget. A tag written at another budget is a tag
# `collect.sh` reads as this study's and the card never defined.
cf412_require_head_steps(){  # <head steps>
  [ "${1:-}" = "$CF412_HEAD_STEPS" ] && return 0
  echo "ABORT: head steps='${1:-}' is not this card's budget" \
       "($CF412_HEAD_STEPS)" >&2
  return 2
}
