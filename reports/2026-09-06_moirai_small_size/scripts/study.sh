#!/bin/bash
# #412 — one cell, six published configurations, at the size of a small
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
# `run_arm.sh` reads it back off the trainer's own command line. A leg that
# does not name its objective in its first CF412_CHECK_TIMEOUT seconds stops
# too: an unchecked leg is not a checked one.
#
# ---- The second thing that loses a leg in silence ----------------------------
#
# A backbone can lose the contrastive task. #404 saw one at seed 20260521. The
# two decay arms carry the higher risk, because `L_rep` holds the negatives of
# this objective and their weight reaches 0.0 at step 2,000. One k = 32 arm
# costs 32 GPU-hours to 200,000 steps, so `auc_guard.sh` reads the trainer's
# own `auc` column while the leg runs and stops the arm that lost the task.
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
# The leg runner. A variable so a test can hand `run_arm.sh` a stub and prove
# the guards fire. The study never sets it.
CF412_RUNNER="${CF412_RUNNER:-$CF412_PARENT/scripts/run_leg_k.sh}"
# The `L_rep` decay, as the card states it: weight 1.0 at step 0, 0.0 at the
# arm's ramp. The ramp is the arm's column, and it is `-` for an arm that
# carries no decay.
CF412_REP_W_START="${CF412_REP_W_START:-1.0}"
CF412_REP_W_END="${CF412_REP_W_END:-0.0}"
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
  # The AUC gate sleeps between reads, and a 600-second sleep outlives a leg
  # of a few hundred steps. The WARM-UP does not scale: a run of a few hundred
  # steps has not learned the task yet, so the watch must give no verdict on
  # it, and 1,000 steps is what keeps it silent.
  CF412_AUC_POLL="${CF412_AUC_POLL:-10}"
fi

# The first pass of the plan, and the default stop of `phase1.sh`. Every arm at
# every stop is 164 GPU-hours of backbone on one card, so no default asks for
# it. The gate on the 40,000-step scores picks the arms that climb (`run.sh`).
# A trial holds one stop, and this is that stop.
CF412_FIRST_STOP="${CF412_STOPS%% *}"

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

# One arm's row, as `<arm> <k> <reduce> <tau> <end> <ramp> <seed> <decay>`.
# Prints nothing, and returns non-zero, for an arm the table does not hold.
cf412_arm_row(){  # <arm>
  awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $1, $2, $3, $4, $5, $6, $7, $8; found = 1 }
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
  local row name k red tau end ramp seed decay
  row="$(cf412_arm_row "${1:?arm}")" || return 1
  read -r name k red tau end ramp seed decay <<<"$row"
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

# ---- The L_rep decay ---------------------------------------------------------
#
# Configurations 4 and 5 decay the weight on `L_rep` from 1.0 to 0.0 by step
# 2,000. `L_rep` carries the negatives of this objective, so at 0.0 nothing
# pushes the representations apart. #409 built the three trainer flags and
# measured the decay at 1.1M parameters, where it never won. This card asks the
# same question at 11.4M, and configuration 5 minus configuration 2 answers it
# directly.

# The decay ramp of one arm, in steps, or `-` for an arm with no decay. This is
# a FACT about the arm, so no environment value moves it: the tables and the
# figures read it.
cf412_decay_ramp(){  # <arm>
  local v
  v="$(cf412_arm_row "${1:?arm}" | awk '{print $8}')" || return 1
  printf '%s\n' "${v:--}"
}

# The ramp ONE LEG runs, in steps. A trial scales it by the trial budget, so a
# 400-step smoke still crosses the whole decay and its `rep_w` column still
# reaches 0.0. Prints nothing for an arm with no decay.
cf412_ramp(){  # <arm>
  local ramp
  ramp="$(cf412_decay_ramp "${1:?arm}")" || return 1
  [ "$ramp" = "-" ] && return 0
  if [ -n "${CF412_TRIAL:-}" ]; then
    ramp=$(( ramp * CF412_TRIAL / 40000 ))
    [ "$ramp" -ge 1 ] || ramp=1
  fi
  printf '%s\n' "$ramp"
}

# The trainer flags of one arm's decay, as ONE unit. Empty for an arm with no
# decay, which is then byte-for-byte the objective of its plain twin.
cf412_decay_args(){  # <arm>
  local ramp
  ramp="$(cf412_ramp "${1:?arm}")" || return 1
  [ -n "$ramp" ] || return 0
  printf -- '--rep-loss-weight %s --rep-loss-weight-end %s --rep-loss-weight-ramp-steps %s\n' \
    "$CF412_REP_W_START" "$CF412_REP_W_END" "$ramp"
}

# The same decay, in the shape a command line reads back as: `<start> <end>
# <ramp>`, with `-` for a flag the line does not carry. train.py's own default
# start is 1.0, so an arm with no decay reads `1.0 - -`.
cf412_decay_sig(){  # <arm>
  local ramp
  ramp="$(cf412_ramp "${1:?arm}")" || return 1
  [ -n "$ramp" ] || { printf '%s - -\n' "$CF412_REP_W_START"; return 0; }
  printf '%s %s %s\n' "$CF412_REP_W_START" "$CF412_REP_W_END" "$ramp"
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

# The decay a command line names, in the shape of `cf412_decay_sig`.
cf412_decay_of_cmdline(){
  local line start end ramp
  line="$(cat)"
  start="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --rep-loss-weight)"
  end="$(printf '%s' "$line" | cf412_last_arg_of_cmdline --rep-loss-weight-end)"
  ramp="$(printf '%s' "$line" \
    | cf412_last_arg_of_cmdline --rep-loss-weight-ramp-steps)"
  printf '%s %s %s\n' "${start:-$CF412_REP_W_START}" "${end:--}" "${ramp:--}"
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

# ---- The losses CSV of a leg -------------------------------------------------

# The CSVs of one leg, oldest first. A leg re-fired after a crash resumes into
# the SAME leg directory, and train.py branches its `--run-name` to `<name>_r2`
# when that directory already holds `<name>_*.pth` (`safe_run_name`). So one
# arm can hold more than one CSV, and the report reads them all.
cf412_losses_csvs(){  # <arm> <stop steps>
  local dir name
  dir="$(cf412_leg_dir "${1:?arm}" "${2:?stop}")"
  name="$(cf412_run_name "$1")"
  ls -tr "$dir/$name"_losses.csv "$dir/$name"_r[0-9]*_losses.csv 2>/dev/null
}

# The CSV the leg that runs NOW writes to, which is the newest one. The AUC
# gate reads this one: an older CSV holds the steps of a leg that already
# stopped, and a verdict on those would stop the wrong run.
cf412_live_losses_csv(){  # <arm> <stop steps>
  cf412_losses_csvs "${1:?arm}" "${2:?stop}" | tail -1
}

# How many DATA rows a losses CSV holds. The header does not count, and a file
# that is missing or empty holds none.
cf412_csv_rows(){  # <csv>
  awk 'END { n = NR - 1; if (n < 0) n = 0; print n }' "${1:?csv}" 2>/dev/null \
    || printf '0\n'
}

# ---- The AUC gate ------------------------------------------------------------
#
# The card asks for the contrastive AUC of every run, and it names the reading:
# a rolling median over 500 rows against 0.55, after a 1,000-step warm-up. A
# run is lost when that median ends under the threshold and does not come back.
#
# `scripts/auc_watch.py` of the main checkout is the reader, and #409 wrote it.
# `auc_guard.sh` runs it against the live CSV while a leg trains and stops the
# arm that lost the task. A k = 32 arm costs 32 GPU-hours to 200,000 steps, so
# a lost arm must not climb in silence.
#
# A stopped arm is a RESULT, not a failure. It has its whole AUC curve and its
# loss by term to the step it reached.
CF412_AUC_WATCH_PY="${CF412_AUC_WATCH_PY:-$CF412_REPO/scripts/auc_watch.py}"
CF412_AUC_WINDOW="${CF412_AUC_WINDOW:-500}"
CF412_AUC_THRESHOLD="${CF412_AUC_THRESHOLD:-0.55}"
# Steps the verdict does not read. The AUC of a fresh run starts near 0.5 and
# climbs, so a gate with no warm-up stops every arm in its first minute.
CF412_AUC_WARMUP="${CF412_AUC_WARMUP:-1000}"
# How often the gate reads the CSV. The trainer flushes every 100 rows.
CF412_AUC_POLL="${CF412_AUC_POLL:-600}"

# What the gate writes when it stops an arm. The report reads the step out of
# it, and `phase1.sh` reads its presence.
cf412_collapse_file(){  # <arm>
  printf '%s/collapsed_%s.txt\n' "$CF412_RESULTS" "${1:?arm}"
}

# ---- The free memory one leg needs -------------------------------------------
#
# Both cards of this box carry other work from other projects, so a leg starts
# only when the card holds room for it. `run_leg_k.sh` has no such gate: it
# calls `gpu_gate`, which is a no-op on a Default-mode card, and elisa runs two
# cells on one 4090 on purpose. Without this wait a leg that starts beside a
# neighbour at its peak dies inside `.to(device)` and loses the steps since its
# last 20,000-step save.
#
# The need is this arm's OWN peak from `results/trial/smoke.csv`, plus a
# margin. #373's head gate reads the same `memory.free` column.
CF412_VRAM_MARGIN_MIB="${CF412_VRAM_MARGIN_MIB:-1200}"
CF412_VRAM_POLL="${CF412_VRAM_POLL:-60}"
CF412_VRAM_TIMEOUT="${CF412_VRAM_TIMEOUT:-86400}"

# The free memory one arm's leg waits for, in MiB. The peaks are the smoke's,
# rounded up: 10,062 MiB at k = 32, 7,160 at k = 8, 6,472 at k = 3.
cf412_leg_vram_mib(){  # <arm>
  local k peak
  k="$(cf412_depth "${1:?arm}")" || return 1
  case "$k" in
    3) peak=6500 ;;
    8) peak=7200 ;;
    *) peak=10100 ;;
  esac
  printf '%s\n' "$(( peak + CF412_VRAM_MARGIN_MIB ))"
}

# Wait until one card reports <need> MiB free. Returns 0 when it does, 1 on
# the timeout. A card whose `memory.free` cannot be read returns 0: a box
# without `nvidia-smi` must run, not block.
cf412_wait_for_vram(){  # <gpu index> <need MiB> [label]
  local gpu="${1:?gpu}" need="${2:?need}" label="${3:-leg}" waited=0 free
  while :; do
    free="$(nvidia-smi --id="$gpu" --query-gpu=memory.free \
              --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    [ -n "$free" ] || return 0
    [ "$free" -ge "$need" ] && {
      [ "$waited" -gt 0 ] && echo "[#412 vram] gpu $gpu has ${free} MiB free" \
        "after ${waited}s — $label starts"
      return 0; }
    if [ "$waited" -ge "$CF412_VRAM_TIMEOUT" ]; then
      echo "[#412 vram] TIMEOUT after ${waited}s: gpu $gpu has ${free} MiB" \
        "free, $label needs ${need}" >&2
      return 1
    fi
    [ $(( waited % 600 )) -eq 0 ] && \
      echo "[#412 vram] gpu $gpu has ${free} MiB free, $label needs ${need}"
    sleep "$CF412_VRAM_POLL"; waited=$(( waited + CF412_VRAM_POLL ))
  done
}

# ---- The smoke ---------------------------------------------------------------
#
# The step time and the peak memory the run plan is sized from. A 5-second
# sampler over a 25-second arm can miss the peak, so the poll is one second and
# each arm runs long enough that the stream is warm before the timing line the
# table reads.
CF412_SMOKE_STEPS="${CF412_SMOKE_STEPS:-150}"
CF412_SMOKE_POLL="${CF412_SMOKE_POLL:-1}"

# ---- Exit codes --------------------------------------------------------------
#
#   2   refused: not an arm, not a stop, no runner, no checkpoint
#   3   the trainer took an objective this arm does not carry, or it named none
#   4   the AUC gate stopped this arm
#   9   the session holds above this stop (`run_leg_k.sh`)
#   10  another machine claims this cell (`run_leg_k.sh`)
CF412_RC_COLLAPSED=4

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
