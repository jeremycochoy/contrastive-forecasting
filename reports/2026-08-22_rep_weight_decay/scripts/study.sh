#!/bin/bash
# #409 — a decay of the L_rep weight to zero, over an EMA schedule and a ramp.
# Sourced, never run.
#
# The cell is #373's `arm6_v2_combab_alignT` at k = 32 under the `mean`
# reduction: `cosine_similarity_batch_rep_only`, L_align on the EMA TEACHER at
# weight 1.0, MoCo rep keys, tau_rep = 1.0, no CPC auxiliary, SIGReg on the
# embedding and the encoding at weight 1.0, tau = 0.10.
#
# The decay is one extra factor in front of L_rep. It starts at 1.0 and falls
# linearly to 0.0 at the arm's RAMP. The two ends are the card's, so they live
# here. The RAMP is a column of the arms table: four rows hold one schedule and
# one seed and differ in the ramp alone.
#
# ---- Why the weight ----------------------------------------------------------
#
# For this loss shape L_rep IS the whole main loss. It holds 92 to 93 percent of
# the total at step 40,000 and reaches its level near step 100. The term that
# moves after step 500 is L_align. So most of the objective is a term that
# stopped moving, and the card asks what happens when it is decayed out.
#
# ---- The axis is the EMA schedule --------------------------------------------
#
# The align target is the teacher. Past step 10,000 the decay has taken L_rep
# out, so L_align is the whole main loss and the EMA teacher is the only target
# left. The schedule sets how fast that teacher moves, so it keeps acting for
# three quarters of the run.
#
# The card allows eight backbones, and the search spends them one round at a
# time. Each round reads the scores of the round before it, so
# `scripts/arms.tsv` is a CATALOGUE of candidates and not a queue. Some of its
# rows never ran. `notes/search_protocol.md` holds the rule.
#
# `EMA_ARGS` in #373's runner REPLACES the three momentum flags, which is what
# a fixed arm needs: it passes `--ema-tau` alone, and no repeated flag can
# remove `--ema-tau-end`. See scripts/arms.tsv for every row and for why each
# one earns a backbone.
#
# ---- This card runs no control -----------------------------------------------
#
# The sweep scored seven of the eight schedules on this same cell with NO
# decay, at the same stop and the same head, and
# `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md` holds them. Those
# are the references. A control arm here would spend a backbone on a number
# that exists.
#
# ---- What this study does not write ------------------------------------------
#
# The card compares its arms to published #404 numbers on this same cell. A
# path shared with #373 or #404 overwrites one of them, so four names carry the
# arm: the checkpoint root, the run name, the tag and the score file.

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
CF409_CELL="arm6_v2_combab_alignT"
CF409_K=32
CF409_REDUCE="mean"
# The cell's own L_align target. It is stated on every leg, so a leg log names
# the objective it trained instead of leaving a reader to infer the cell's
# default.
CF409_ALIGN_TARGET="teacher"
# One stop. The card trains each arm to 40,000 backbone steps, because that is
# where the sweep measured this cell.
CF409_STOPS="40000"
CF409_HEAD_STEPS=30000
# How often a leg writes a step checkpoint. #373's runner defaults to 20,000,
# which is half of this card's whole stop: on 08-23 `dec_m080_r200` reached
# 19,900 steps, saved nothing and lost all of it to a DNS outage. At 5,000 an
# outage costs at most 5,000 steps, and eight checkpoint pairs cost 88 MB.
CF409_SAVE_EVERY="${CF409_SAVE_EVERY:-5000}"
CF409_HEAD_SEED=20260722
CF409_ENC="student"

# ---- The decay, which is the card's whole change -----------------------------
#
# The weight starts at 1.0 and falls linearly to 0.0 at the arm's ramp, then
# holds. The two ends are the card's, so they live here.
CF409_REP_W_START="${CF409_REP_W_START:-1.0}"
CF409_REP_W_END="${CF409_REP_W_END:-0.0}"
# The ramp of an ARM is column 5 of its row, and `cf409_decay_ramp_of` reads
# it. The card's own ramp is 10,000 steps, and `scripts/arms.tsv` states it
# where a reader writes a new row.
#
# This is an OVERRIDE of that column, for one dry run of a ramp no row carries
# yet. It moves the LEG only. It never moves `cf409_decay_ramp_of`, because a
# stray value in a lane environment would otherwise rewrite the ramp column of
# every arm in `results/scores.csv`.
CF409_REP_W_RAMP="${CF409_REP_W_RAMP:-}"

# ---- The latent-drift probe --------------------------------------------------
#
# The probe is a DIAGNOSTIC. It draws a fixed ARMA batch once, then does one
# no-grad forward of it at every save step and writes the drift of h_t to
# `<run>_latent_drift.csv`. At the trainer's own batch of 64 that forward
# allocates a block of several GB, and the allocator keeps it for the run. On
# the k = 3 cell that block, not the training, set what one leg held.
#
# elisa's two cards already carry other agents' runs, and this card must share
# them. At 16 the block is about a quarter of that, which leaves the training
# itself as the larger half.
#
# The probe cannot move the training. `generate_arma_batch` draws its batch
# from `np.random.default_rng(seed)`, which is a LOCAL generator, and `probe()`
# runs under `torch.no_grad()` on a model it puts back in train mode. So this
# changes the drift CSV of every arm and nothing else.
#
# Every arm of this card takes the same value, so no arm's drift CSV is on a
# different footing from another's.
CF409_PROBE_BS="${CF409_PROBE_BS:-16}"
CF409_ARMS_TSV="${CF409_ARMS_TSV:-$CF409_SCRIPTS/arms.tsv}"

# ---- Trial mode --------------------------------------------------------------
# `CF409_TRIAL=<backbone steps>` runs the WHOLE pipeline at a budget that
# finishes in minutes: the same wrappers, the same #373 runner, the same head
# script and the same guards. One arm to 40,000 steps takes hours, so a trial
# puts the first wiring defect minutes from the start.
#
#   CF409_TRIAL=400 bash scripts/run_arm.sh dec_s20 400
#
# `cf409_ramp` scales each arm's own ramp by the same fraction, so every arm of
# a trial still crosses its whole decay.
if [ -n "${CF409_TRIAL:-}" ]; then
  CF409_STOPS="$CF409_TRIAL"
  CF409_HEAD_STEPS=$(( CF409_TRIAL / 2 ))
  # The AUC gate scales too, or a 400-step trial would end inside its own
  # warmup and the gate would never fire.
  CF409_AUC_WARMUP="${CF409_AUC_WARMUP:-$(( CF409_TRIAL / 40 ))}"
  CF409_AUC_POLL="${CF409_AUC_POLL:-30}"
fi

# ---- Where the artefacts live ------------------------------------------------
#
# The durable root. Never /tmp, never inside the checkout (CLAUDE.md
# checkpoint safety rule 4), and never #373's or #404's root — one root for
# two studies is a sync loop that cannot tell their checkpoints apart.
#
# This card trains on elisa, so this root is on elisa's own disk and no sync
# loop runs. `notes/artefacts.md` names every artefact and its path.
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
  awk -F'\t' '!/^#/ && NF >= 6 { print $1 }' "$CF409_ARMS_TSV"
}
CF409_ARMS="$(cf409_arms | tr '\n' ' ')"
CF409_ARMS="${CF409_ARMS% }"

# One arm's row, as `<arm> <tau> <end> <ema ramp> <decay ramp> <seed>`. Prints
# nothing, and returns non-zero, for an arm the table does not hold.
cf409_arm_row(){  # <arm>
  awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $1, $2, $3, $4, $5, $6; found = 1 }
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

# The backbone seed of one arm, from column 6. A seed that does not reach the
# trainer moves an arm onto another arm's data order. A row that moves the seed
# says so in its name, because a repeat seed measures the error bar of the arm
# it repeats.
cf409_seed(){  # <arm>
  cf409_arm_row "${1:?arm}" | awk '{print $6}'
}

# ---- The EMA schedule, which is this card's axis ------------------------------

# The trainer flags of one arm's momentum, as ONE unit.
#
# A fixed arm passes `--ema-tau` alone. It does NOT pass `--ema-tau-end` at any
# value: train.py reads "no end value" as a constant momentum, and no value of
# the flag means the same. This is why these flags REPLACE the runner's
# schedule through `EMA_ARGS` rather than ride the appended `GAP_ARGS` block —
# a repeat can change a flag, never remove one.
cf409_ema_args(){  # <arm>
  local row name tau end ramp rep_ramp seed
  row="$(cf409_arm_row "${1:?arm}")" || return 1
  read -r name tau end ramp rep_ramp seed <<<"$row"
  if [ "$end" = "-" ]; then
    printf -- '--ema-tau %s\n' "$tau"
  else
    printf -- '--ema-tau %s --ema-tau-end %s --ema-tau-ramp-steps %s\n' \
      "$tau" "$end" "$ramp"
  fi
}

# The same three values, in the shape of an arms.tsv row: `<tau> <end> <ramp>`,
# with `-` for a flag the arm does not pass. So a leg log compares against the
# table with one string equality, not three.
cf409_ema_sig(){  # <arm>
  cf409_arm_row "${1:?arm}" | awk '{print $2, $3, $4}'
}

# The momentum an arm HOLDS at a given step, which is not the momentum its
# command line names. This is the number that ranks two arms at one stop:
# `dec_m090_r60` and `dec_m090_r200` both name 0.9 and hold 0.967 and 0.920 at
# 40,000 steps.
#
# The formula is `src.models.ema_tau_at_step`, which is linear and clamps the
# step into the ramp. It is repeated here, and not imported, because the shell
# readers of this study must not need a Python interpreter to print a table.
# `tests/test_409_launcher_shape.py` holds the two against each other.
cf409_momentum_at(){  # <arm> <step>
  local row name tau end ramp rep_ramp seed
  row="$(cf409_arm_row "${1:?arm}")" || return 1
  read -r name tau end ramp rep_ramp seed <<<"$row"
  if [ "$end" = "-" ]; then printf '%.3f\n' "$tau"; return 0; fi
  awk -v t="$tau" -v e="$end" -v r="$ramp" -v s="${2:?step}" 'BEGIN{
    if (r + 0 <= 0) { printf "%.3f\n", e; exit }
    f = s / r; if (f > 1) f = 1; if (f < 0) f = 0;
    printf "%.3f\n", t + f * (e - t) }'
}

# One arm's schedule, as a reader sees it. `0.9 fixed` or `0.9 to 1.0 at 100k`.
cf409_ema_label(){  # <arm>
  local row name tau end ramp rep_ramp seed
  row="$(cf409_arm_row "${1:?arm}")" || return 1
  read -r name tau end ramp rep_ramp seed <<<"$row"
  if [ "$end" = "-" ]; then printf '%s fixed\n' "$tau"
  else printf '%s to %s at %s\n' "$tau" "$end" "$(cf409_steps_label "$ramp")"
  fi
}

# ---- The decay ---------------------------------------------------------------

# The decay ramp of one arm, in steps, from column 5 of its row. This is a FACT
# about the arm, so no environment value moves it: `results/scores.csv`, the
# figures and `results/RUN_STATE.md` all read it.
cf409_decay_ramp_of(){  # <arm>
  cf409_arm_row "${1:?arm}" | awk '{print $5}'
}

# The ramp ONE LEG of an arm runs, in steps. The row gives it.
# `CF409_REP_W_RAMP` replaces it, which is how a dry run tries a ramp that has
# no row yet. A trial scales it by the trial budget, so a 400-step trial still
# crosses its whole decay and its `rep_w` column still reaches 0.0.
cf409_ramp(){  # <arm>
  local ramp
  ramp="${CF409_REP_W_RAMP:-$(cf409_decay_ramp_of "${1:?arm}")}" || return 1
  [ -n "$ramp" ] || return 1
  if [ -n "${CF409_TRIAL:-}" ]; then
    ramp=$(( ramp * CF409_TRIAL / 40000 ))
    [ "$ramp" -ge 1 ] || ramp=1
  fi
  printf '%s\n' "$ramp"
}

# The trainer flags of the decay of one arm, as ONE unit.
cf409_decay_args(){  # <arm>
  printf -- '--rep-loss-weight %s --rep-loss-weight-end %s --rep-loss-weight-ramp-steps %s\n' \
    "$CF409_REP_W_START" "$CF409_REP_W_END" "$(cf409_ramp "${1:?arm}")"
}

# The same three values as the trainer's own command line reports them, so a
# leg's log can be read against the card. `-` for a flag the command line does
# not carry.
cf409_decay_sig(){  # <arm>
  printf '%s %s %s\n' "$CF409_REP_W_START" "$CF409_REP_W_END" \
    "$(cf409_ramp "${1:?arm}")"
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

cf409_reduce_of_cmdline(){
  awk '{ r = "-"
    for (i = 1; i <= NF; i++) if ($i == "--train-rollout-reduce") r = $(i + 1)
    print r }'
}

# The momentum a trainer command line carries, as `<tau> <end> <ramp>`, with
# `-` for a flag the line does not hold. `cf409_ema_sig` gives the same shape
# out of the arms table, so `run_arm.sh` compares the two.
cf409_ema_of_cmdline(){
  awk '{ t = "-"; e = "-"; r = "-"
    for (i = 1; i <= NF; i++) {
      if ($i == "--ema-tau") t = $(i + 1)
      if ($i == "--ema-tau-end") e = $(i + 1)
      if ($i == "--ema-tau-ramp-steps") r = $(i + 1)
    }
    print t, e, r }'
}

# The weight the arms HOLD at a given step, which is not the weight the command
# line names. This is the number that reads one point of training.
#
# The formula is `src.models.linear_schedule_at_step`, which is linear and
# clamps the step into the ramp. It is repeated here, and not imported,
# because the shell readers of this study must not need a Python interpreter
# to print a table. `tests/test_409_launcher_shape.py` holds the two against
# each other.
cf409_rep_w_at(){  # <arm> <step>
  awk -v w="$CF409_REP_W_START" -v e="$CF409_REP_W_END" \
      -v r="$(cf409_ramp "${1:?arm}")" -v s="${2:?step}" 'BEGIN{
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
# lay every one of them into one <root>/<cell>/leg_40k/. The run names differ,
# but a save dir shared by several runs is CLAUDE.md checkpoint safety rule 3.
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

# The CSVs of one leg, oldest first. A leg re-fired after a crash resumes into
# the SAME leg_40k/ directory, and train.py branches its `--run-name` to
# `<name>_r2` when that directory already holds `<name>_*.pth`
# (`safe_run_name`). So one arm can hold more than one CSV, and the report
# reads them all.
cf409_losses_csvs(){  # <arm> <stop steps>
  local dir name
  dir="$(cf409_leg_dir "${1:?arm}" "${2:?stop}")"
  name="$(cf409_run_name "$1")"
  ls -tr "$dir/$name"_losses.csv "$dir/$name"_r[0-9]*_losses.csv 2>/dev/null
}

# The CSV the leg that runs NOW writes to, which is the newest one. The AUC
# gate reads this one: an older CSV holds the steps of a leg that already
# stopped, and a verdict on those would stop the wrong run.
cf409_live_losses_csv(){  # <arm> <stop steps>
  cf409_losses_csvs "${1:?arm}" "${2:?stop}" | tail -1
}

# How many DATA rows a losses CSV holds. The header does not count, and a file
# that is missing or empty holds none.
#
# The AUC gate counts these before a leg starts. A leg that resumes with no
# checkpoint on disk keeps its run name, and train.py then APPENDS to the CSV
# of the leg that crashed — so "the newest CSV" alone does not say which rows
# this leg wrote. The count does.
cf409_csv_rows(){  # <csv>
  awk 'END { n = NR - 1; if (n < 0) n = 0; print n }' "${1:?csv}" 2>/dev/null \
    || printf '0\n'
}

# ---- The AUC gate ------------------------------------------------------------
#
# L_rep carries the negatives of this objective. Every arm decays it to 0.0 by
# step 10,000, and past that step nothing pushes the representations apart. An
# arm that lost the contrastive task then climbs about 30,000 dead steps to a
# stop whose score is already known to be bad.
#
# `auc_guard.sh` reads the `auc` column of the live CSV while the leg trains and
# stops the leg that lost the task. The window and the threshold are
# `auc_watch.py`'s own: a 500-step rolling median, under 0.55. The sweep's one
# collapsed backbone sat at 0.5745 at the stop.
#
# A stopped arm is a RESULT, not a failure. The card asks whether any run loses
# the contrastive task, and at which step.
CF409_AUC_WINDOW="${CF409_AUC_WINDOW:-500}"
CF409_AUC_THRESHOLD="${CF409_AUC_THRESHOLD:-0.55}"
# Steps the verdict does not read. The AUC of a fresh run starts near 0.5 and
# climbs, so a gate with no warmup stops every arm in its first minute. Every
# arm holds a weight of 0.9 or more through this warmup, so no arm can collapse
# from the decay inside it.
CF409_AUC_WARMUP="${CF409_AUC_WARMUP:-1000}"
# How often the gate reads the CSV. The trainer flushes every 100 rows.
CF409_AUC_POLL="${CF409_AUC_POLL:-600}"

# What the gate writes when it stops an arm. The report reads the step out of
# it, and `phase1.sh` reads its presence.
cf409_collapse_file(){  # <arm>
  printf '%s/collapsed_%s.txt\n' "$CF409_RESULTS" "${1:?arm}"
}

# ---- Exit codes --------------------------------------------------------------
#
# A lane re-fires a leg that CRASHED. It must never re-fire a refusal, because
# a refusal repeats and each re-fire costs a GPU lane.
#
#   2   refused: not an arm, not a stop, no runner, no checkpoint
#   3   the trainer took an objective this arm does not carry
#   4   the AUC gate stopped this arm
#   9   the session holds above this stop (`run_leg_k.sh`)
#   10  another machine claims this cell (`run_leg_k.sh`)
CF409_RC_COLLAPSED=4
CF409_NO_RETRY="2 3 4 9 10"

# ---- The Hub outage of 2026-08-23 --------------------------------------------
#
# The data streams from the Hub, so a box that loses DNS kills every leg in
# about 3 seconds. That day the lane read three of those as failed arms, spent
# each arm's ladder in two minutes and left the card idle for 27 hours.
#
# `scripts/hub_gate.sh` holds the reading of a dead leg's tail and the wait.
# Code 20 is what a leg gives when the HUB failed. It is NOT in
# `CF409_NO_RETRY`, and `phase1.sh` does not count it against the ladder
# either: the arm is fine, so the try was not the arm's.
. "$CF409_REPO/scripts/hub_gate.sh"
CF409_RC_NETWORK="$HUB_GATE_RC"
# How long a lane waits for the Hub, in total. Hours, not minutes.
CF409_NET_DEADLINE="${CF409_NET_DEADLINE:-$HUB_GATE_DEADLINE}"

cf409_retryable(){  # <exit code>
  [ "${1:?rc}" -eq 0 ] && return 1
  case " $CF409_NO_RETRY " in *" $1 "*) return 1 ;; esac
  return 0
}

# How many times a lane fires ONE leg, and ONE head, before it drops it.
CF409_LEG_TRIES="${CF409_LEG_TRIES:-3}"
CF409_HEAD_TRIES="${CF409_HEAD_TRIES:-3}"

# ---- The head half of the study ----------------------------------------------
#
# One 30,000-step student head on each arm's 40,000-step backbone, then that
# head's 97 GIFT-Eval configs. `head_eval.sh` runs the pair on #373's protocol,
# which is also the sweep's, so the reference scores compare. The names below
# are what the pair writes under.

cf409_is_in(){  # <value> <space separated list>
  case " ${2:-} " in *" ${1:-} "*) return 0 ;; *) return 1 ;; esac
}

# The card defines ONE head budget. A tag written at any other budget is a tag
# `collect.sh` reads as this study's and the card never defined.
cf409_require_head_steps(){  # <head steps>
  [ "${1:-}" = "$CF409_HEAD_STEPS" ] && return 0
  echo "ABORT: head steps='${1:-}' is not this card's budget" \
       "($CF409_HEAD_STEPS)" >&2
  return 2
}

# A step count, as a tag reads it. `40000` -> `40k`, `400` -> `400`. A trial
# budget is not a multiple of 1000, and rounding it to `0k` would give two
# budgets one tag.
cf409_steps_label(){  # <steps>
  local n="${1:?steps}"
  if [ $(( n % 1000 )) -eq 0 ]; then printf '%dk' $(( n / 1000 ))
  else printf '%d' "$n"; fi
}

cf409_steps_of(){  # <label>
  case "${1:?label}" in
    *k) printf '%d' $(( ${1%k} * 1000 )) ;;
    *)  printf '%d' "$1" ;;
  esac
}

# The name of one (arm, stop, head budget). It names the head checkpoint, the
# eval directory and the score file, so `collect.sh` reads the table back out
# of the filenames rather than out of a second place that can drift.
cf409_tag(){  # <arm> <stop steps> <head steps>
  printf '%s_bb%s_h%s_%s\n' "${1:?arm}" \
    "$(cf409_steps_label "${2:?stop}")" \
    "$(cf409_steps_label "${3:?head steps}")" "$CF409_ENC"
}

# The backbone a stop produced, or nothing. Two names, not one glob:
# `<name>_<N>k.pth` is the leg's own, and `<name>_r<N>_<N>k.pth` is train.py's
# `_rN` infix on a re-fired leg. A trailing `*` would take the optimizer file
# with them.
#
# train.py names every snapshot `<run>_<step / 1000>k.pth`, so a trial budget
# of 400 steps lands `_0k.pth`. The TAG uses `cf409_steps_label` instead, which
# keeps a trial's score file apart from the study's.
cf409_bb_ckpt(){  # <arm> <stop steps>
  local dir name kk
  dir="$(cf409_leg_dir "${1:?arm}" "${2:?stop}")"
  name="$(cf409_run_name "$1")"
  kk=$(( ${2:?stop} / 1000 ))
  ls "$dir/$name"_"$kk"k.pth "$dir/$name"_r[0-9]*_"$kk"k.pth 2>/dev/null \
    | grep -v optimizer | head -1
}

# Where #373's head script lays one tag's head, its GIFT-Eval output and its
# merged 97-config CSV. One root per arm, the same rule as the backbones.
cf409_eval_dir(){  # <arm> <tag>
  printf '%s/eval/%s\n' "$(cf409_arm_root "${1:?arm}")" "${2:?tag}"
}

# The file #373's head script writes one pair's aggregate score to.
cf409_score_file(){  # <arm> <stop steps>
  printf '%s/score_%s.txt\n' "$CF409_RESULTS" \
    "$(cf409_tag "${1:?arm}" "${2:?stop}" "$CF409_HEAD_STEPS")"
}

# Every (arm, stop) pair of the study, one per line.
cf409_pairs(){
  local arm stop
  for arm in $CF409_ARMS; do
    for stop in $CF409_STOPS; do printf '%s %s\n' "$arm" "$stop"; done
  done
}

cf409_scored(){  # <arm> <stop steps>
  [ -s "$(cf409_score_file "${1:?arm}" "${2:?stop}")" ]
}

# ---- The cards this machine carries ------------------------------------------
#
# elisa holds two RTX 4090s and the study deals its arms over them. A lane on a
# card that is not there dies inside `.to(device)`, hours after the operator
# has left, so the launcher asks the driver first.
#
# `CF409_GPU_COUNT` overrides the count, for a test that must not depend on the
# machine it runs on.
cf409_gpu_count(){
  if [ -n "${CF409_GPU_COUNT:-}" ]; then printf '%s\n' "$CF409_GPU_COUNT"
    return 0; fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .
}

cf409_default_gpus(){
  local n i out=""
  n="$(cf409_gpu_count)"
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  for (( i = 0; i < n; i++ )); do out="$out $i"; done
  printf '%s\n' "${out# }"
}

cf409_require_gpus(){  # <space separated indices>
  local n g bad=0
  n="$(cf409_gpu_count)"
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  if [ "$n" -lt 1 ]; then
    echo "ABORT: this machine carries no card, so no lane can start" >&2
    return 2
  fi
  for g in ${1:-}; do
    case "$g" in
      ''|*[!0-9]*)
        echo "ABORT: gpu '$g' is not a card index" >&2; bad=1; continue ;;
    esac
    [ "$g" -lt "$n" ] && continue
    echo "ABORT: gpu $g — this machine carries $n card(s), so the indices" >&2
    echo "  are 0 to $(( n - 1 ))." >&2
    bad=1
  done
  [ "$bad" -eq 0 ]
}

# ---- The checkout this study needs -------------------------------------------
#
# Four things this card depends on are NOT in every checkout of this
# repository. A machine bootstrapped from a stale branch would train, log
# nothing unusual, and hand back one copy of the published cell for each arm —
# or a set of backbones and an empty scores.csv.
#
#   --rep-loss-weight-end in the trainer (#409). Without it every arm holds the
#   weight at 1.0, so every arm repeats a number the sweep already published.
#
#   GAP_ARGS in #373's runner. Without it the decay and the seed never reach
#   the trainer. `run_arm.sh` also catches this at run time, off the trainer's
#   own command line. This check is the cheap one, before the first leg.
#
#   EMA_ARGS in #373's runner (#404). The schedule is this card's AXIS. Without
#   it every arm trains the runner's own schedule, so the eight arms would be
#   arm 1 eight times under eight names.
#
#   The head path: #373's `head_eval_bb.sh`, which must read CF_RESULTS, and
#   the GIFT-Eval head trainer it runs. Both refuse on their own, but only
#   AFTER the backbone — so a missing one costs the arm's hours and gives no
#   score.
#
#   The HF token. Every run that streams from HF must authenticate, or the
#   anonymous rate limit idles the card at about 20 percent use.
#
# Prints what is missing and returns non-zero.
cf409_check_checkout(){  # [checkout]
  local wt="${1:-$CF409_WT}" missing=0 runner trainer token head head_train
  runner="$wt/reports/2026-08-08_rollout_depth/scripts/run_leg_k.sh"
  trainer="$wt/experiments/2026-04-27_freq-embedding/scripts/train.py"
  head="$wt/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh"
  head_train="$wt/experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py"
  token="$wt/experiments/hf_token.txt"
  if ! grep -q -- '--rep-loss-weight-end' "$trainer" 2>/dev/null; then
    echo "ABORT: $trainer has no --rep-loss-weight-end." >&2
    echo "  Every arm would hold the weight at 1.0, so every arm would" >&2
    echo "  repeat a number the sweep already published." >&2
    missing=1
  fi
  if ! grep -q 'GAP_ARGS' "$runner" 2>/dev/null; then
    echo "ABORT: $runner takes no GAP_ARGS." >&2
    echo "  The decay and the seed would not reach the trainer." >&2
    missing=1
  fi
  if ! grep -q 'EMA_ARGS' "$runner" 2>/dev/null; then
    echo "ABORT: $runner takes no EMA_ARGS." >&2
    echo "  Every arm would train the runner's own schedule, so the eight" >&2
    echo "  arms would be arm 1 eight times under eight names." >&2
    missing=1
  fi
  if [ ! -f "$head" ]; then
    echo "ABORT: no head script at $head." >&2
    echo "  Every arm would train a backbone for hours, and then every head" >&2
    echo "  would exit 2 and scores.csv would be empty." >&2
    missing=1
  elif ! grep -q 'CF_RESULTS' "$head" 2>/dev/null; then
    echo "ABORT: $head does not read CF_RESULTS." >&2
    echo "  It would write every score_<tag>.txt under #373's results/," >&2
    echo "  where collect.sh does not look." >&2
    missing=1
  fi
  if [ ! -f "$head_train" ]; then
    echo "ABORT: no GIFT-Eval head trainer at $head_train." >&2
    echo "  The head script refuses without it, after the backbone." >&2
    missing=1
  fi
  if [ ! -s "$token" ]; then
    echo "ABORT: no HF token at $token." >&2
    echo "  The anonymous rate limit throttles the stream and idles the" >&2
    echo "  card. The head trainer refuses an empty token outright." >&2
    missing=1
  fi
  [ "$missing" -eq 0 ]
}
