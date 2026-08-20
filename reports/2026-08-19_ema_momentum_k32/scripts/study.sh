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
# The card's backbone seed. One arm of this study is a REPEAT of another at a
# SECOND seed, so the seed is a column of the arms table and this is only the
# value a row that names none takes.
CF404_SEED_DEFAULT="${CF404_SEED_DEFAULT:-20260520}"
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

# The arms table is the one place this study's arms live, so a missing file
# is a study with no arms — and every guard below would then refuse every arm
# with a message about the arm rather than about the file.
[ -f "$CF404_ARMS_TSV" ] || {
  echo "ABORT: no arms table at $CF404_ARMS_TSV" >&2
  return 2 2>/dev/null || exit 2; }

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

# How long the ramp is, in steps. `0` for an arm that holds alpha fixed.
#
# The momentum figure orders the arms of one alpha by this number, so a fixed
# arm, #401's 100,000-step ramp and this card's 200,000-step ramp read left to
# right under one tick instead of landing on one another.
cf404_ramp(){  # <arm>
  local v
  v="$(cf404_arm_row "${1:?arm}" | awk '{print $4}')" || return 1
  case "$v" in ""|-) printf '0\n' ;; *) printf '%s\n' "$v" ;; esac
}

# The backbone seed of one arm, from column 5.
#
# The four arms of the card share one seed, so nothing in the study ever moved
# it. The repeat arm does: it trains the momentum of `s08` a second time at a
# second seed, and the pair is the only thing that measures this cell's own
# run-to-run spread. A row with no fifth column takes the card's seed.
cf404_seed(){  # <arm>
  local v
  v="$(awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $5; found = 1 } END { exit !found }' \
    "$CF404_ARMS_TSV")" || return 1
  case "$v" in ''|-) printf '%s\n' "$CF404_SEED_DEFAULT" ;;
               *) printf '%s\n' "$v" ;; esac
}

# The L_align weight of one arm, from column 6.
#
# WHY THIS COLUMN EXISTS. For this loss shape the rollout depth touches the
# align term alone. A depth copy of the repel term is h-anchored, so it has
# nothing to substitute and adds exactly zero, and the align add-on IS
# duplicated (src/loss.py). The reduction is a mean, so k + 1 copies of
# L_align average back to about one copy's magnitude. The loss then holds ONE
# copy of the h-anchored repel term against the MEAN of k + 1 copies of the
# f-anchored pull term, and --align-loss-weight is the only flag that sets
# that balance.
#
# A `-` or an absent column takes the cell's own value, which run_leg_k.sh
# already puts on every command line. So every arm of rows 1 to 10 keeps the
# command line it ran, and only an arm that names a weight changes one.
CF404_ALIGN_W_DEFAULT="${CF404_ALIGN_W_DEFAULT:-1.0}"
cf404_align_weight(){  # <arm>
  local v
  v="$(awk -F'\t' -v a="${1:?arm}" \
    '!/^#/ && $1 == a { print $6; found = 1 } END { exit !found }' \
    "$CF404_ARMS_TSV")" || return 1
  case "$v" in ''|-) printf '%s\n' "$CF404_ALIGN_W_DEFAULT" ;;
               *) printf '%s\n' "$v" ;; esac
}

# Two numbers, compared as numbers. `3` and `3.0` are one weight, and a table
# that says one while the command line says the other is not a defect.
cf404_num_eq(){  # <a> <b>
  awk -v a="${1:-}" -v b="${2:-}" 'BEGIN{
    if (a == "" || b == "" || a == "-" || b == "-") exit (a != b)
    exit !(a + 0 == b + 0) }'
}

# The trainer flags of one arm's EMA momentum, as ONE unit.
#
# A fixed arm passes `--ema-tau` alone. It does NOT pass `--ema-tau-end` at
# any value: train.py reads "no end value" as "alpha constant", and there is
# no value of the flag that means the same. This is why the flags REPLACE
# #373's schedule (`EMA_ARGS`) rather than appending to it — a repeat can
# change a flag, never remove one.
cf404_ema_args(){  # <arm>
  local row name tau end ramp
  row="$(cf404_arm_row "${1:?arm}")" || return 1
  read -r name tau end ramp <<<"$row"
  if [ "$end" = "-" ]; then
    printf -- '--ema-tau %s\n' "$tau"
  else
    printf -- '--ema-tau %s --ema-tau-end %s --ema-tau-ramp-steps %s\n' \
      "$tau" "$end" "$ramp"
  fi
}

# The momentum an arm HOLDS at a given step, which is not the momentum its
# command line names.
#
# This is the number that compares two arms at one stop. A fixed arm holds the
# value it names. A ramp arm walks from `tau` to `end` over `ramp` steps, so
# the value at the stop depends on the RAMP LENGTH as much as on the start:
# `s08` names 0.8 and holds 0.840 at 40,000 steps over a 200,000-step ramp,
# and `r100_08` names the same 0.8 and holds 0.880 over a 100,000-step ramp.
# An arm named by its start value alone reads as a duplicate of another.
#
# The formula is `src.models.ema_tau_at_step`, which is linear and clamps the
# step into the ramp. It is repeated here, and not imported, because the shell
# readers of this study must not need a Python interpreter to print a table.
# `scripts/test_momentum_at.sh` holds the two against each other.
cf404_momentum_at(){  # <arm> <step>
  local row name tau end ramp step
  row="$(cf404_arm_row "${1:?arm}")" || return 1
  read -r name tau end ramp <<<"$row"
  step="${2:?step}"
  if [ "$end" = "-" ]; then printf '%.3f\n' "$tau"; return 0; fi
  awk -v t="$tau" -v e="$end" -v r="$ramp" -v s="$step" 'BEGIN{
    if (r + 0 <= 0) { printf "%.3f\n", e; exit }
    f = s / r; if (f > 1) f = 1; if (f < 0) f = 0;
    printf "%.3f\n", t + f * (e - t) }'
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

# ---- The head half of the study ----------------------------------------------
#
# One head and one 97-config GIFT-Eval per (arm, stop). `heads_watch.sh` walks
# those pairs as their backbones land, so it asks three questions here: which
# pairs the study defines, which pairs are scored, and which pairs it must
# stop firing.

# How many times the watcher fires ONE pair before it drops it. A head or an
# eval that fails for a stable reason — a bad checkpoint, a missing package, a
# full disk — otherwise re-fires every POLL seconds and holds a GPU lane for
# as long as the session runs.
CF404_HEAD_TRIES="${CF404_HEAD_TRIES:-3}"

# Every (arm, stop) pair, one per line.
cf404_pairs(){
  local arm stop
  for arm in $CF404_ARMS; do
    for stop in $CF404_STOPS; do printf '%s %s\n' "$arm" "$stop"; done
  done
}

cf404_pair_count(){ cf404_pairs | wc -l | tr -d ' '; }

# The file #373's head script writes one pair's aggregate score to.
cf404_score_file(){  # <arm> <stop steps>
  printf '%s/score_%s.txt\n' "$CF404_RESULTS" \
    "$(cf404_tag "${1:?arm}" "${2:?stop}" "$CF404_HEAD_STEPS")"
}

# The attempt counter of one pair. It lives in a file and not in the watcher's
# memory: a watcher restarted after a reboot must not give a head that already
# failed three times three more hours of GPU. Delete the file to try again.
cf404_tries_file(){  # <arm> <stop steps>
  printf '%s/tries_%s.txt\n' "$CF404_RESULTS" \
    "$(cf404_tag "${1:?arm}" "${2:?stop}" "$CF404_HEAD_STEPS")"
}

# Always an integer, including for a counter that does not exist yet, which is
# the normal case on a first pass.
cf404_tries(){  # <arm> <stop steps>
  local n
  n="$(cat "$(cf404_tries_file "${1:?arm}" "${2:?stop}")" 2>/dev/null)"
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  printf '%s\n' "$n"
}

# Count the attempt, and print the new count. The caller counts BEFORE it
# fires the head, so a head that takes the machine down with it still spent a
# try.
cf404_bump_tries(){  # <arm> <stop steps>
  local f n
  f="$(cf404_tries_file "${1:?arm}" "${2:?stop}")"
  n=$(( $(cf404_tries "$1" "$2") + 1 ))
  mkdir -p "$(dirname "$f")"
  printf '%s\n' "$n" >"$f"
  printf '%s\n' "$n"
}

# A pair the watcher must not fire again: it has no score and it used its whole
# try budget. A pair that failed twice and passed on the third try has a score,
# so it is never exhausted.
cf404_exhausted(){  # <arm> <stop steps>
  [ -s "$(cf404_score_file "${1:?arm}" "${2:?stop}")" ] && return 1
  [ "$(cf404_tries "$1" "$2")" -ge "$CF404_HEAD_TRIES" ]
}

cf404_heads_scored(){
  local arm stop n=0
  while read -r arm stop; do
    [ -n "$arm" ] || continue
    [ -s "$(cf404_score_file "$arm" "$stop")" ] && n=$(( n + 1 ))
  done < <(cf404_pairs)
  printf '%s\n' "$n"
}

# The watcher's exit test: no pair is left to fire, because each one is scored
# or each one used its whole try budget.
#
# NOT "one pair is scored". The box hands the four backbones over about five
# hours apart, so at the moment arm 1 is scored the other three are still on
# the box. A watcher that exits there leaves three arms with no head, and
# `launch_elisa.sh`, which waits on it, stops redrawing the figures.
cf404_heads_done(){
  local arm stop
  while read -r arm stop; do
    [ -n "$arm" ] || continue
    [ -s "$(cf404_score_file "$arm" "$stop")" ] && continue
    cf404_exhausted "$arm" "$stop" && continue
    return 1
  done < <(cf404_pairs)
  return 0
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

# The seed a trainer command line names. `-` when it carries no flag, which no
# leg of this study can produce: `run_leg_k.sh` always passes --seed.
cf404_seed_of_cmdline(){
  local v; v="$(cf404_arg_of_cmdline --seed)"
  printf '%s\n' "${v:--}"
}

# The LAST value of a repeated flag, which is the one argparse keeps.
#
# `cf404_arg_of_cmdline` stops at the first hit, and that is right for a flag
# the command line carries once. The align weight is not one: the cell states
# it, and an arm that moves it REPEATS the flag at the end of the line. A
# first-hit reader would report the cell's value on every arm.
cf404_last_arg_of_cmdline(){  # <flag>
  tr '\0 ' '\n\n' | awk -F= -v f="${1:?flag}" '
    $1 == f { if (NF > 1) { v = $2 } else { getline; v = $0 } }
    END { if (v != "") print v }'
}

# The L_align weight a trainer command line names. `-` when it carries no
# flag, which no leg of this study can produce.
cf404_align_of_cmdline(){
  local v; v="$(cf404_last_arg_of_cmdline --align-loss-weight)"
  printf '%s\n' "${v:--}"
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

# A `pgrep -f` pattern that CANNOT match the shell that carries it.
#
# `ssh box "pgrep -f X"` runs `bash -c "pgrep -f X"` ON THE BOX, so the pattern
# X is then inside that shell's own command line. `pgrep -f` reads full command
# lines. It matches the shell, and the check is TRUE on a box with no trainer
# at all. `pgrep` drops its own pid from the result and nothing else, so this
# is invisible in a local test that does not go through a second shell.
#
# On 2026-08-19 `round2_box.sh` sent two bare patterns this way. Three boxes
# read "a trainer already runs on the box", started none, and sat at 0% GPU for
# 30 minutes at $0.3611/h each.
#
# The fix is a bracket class on the first character. `[r]un_leg_k` matches the
# text `run_leg_k` and does NOT match the text `[r]un_leg_k` that the shell
# carries. Send a pattern from here to every remote `pgrep -f`, never a bare
# one. `scripts/test_trainer_check.sh` proves both directions.
cf404_pgrep_pattern(){  # <pattern>
  local p="${1:?pattern}"
  printf '[%s]%s\n' "${p:0:1}" "${p:1}"
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
cf404_sync_loop_pids(){  # <local dir>
  local want="${1:?local dir}" p
  want="${want%/}"
  for p in $(pgrep -f 'sync_loop\.sh' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    tr '\0' '\n' <"/proc/$p/cmdline" 2>/dev/null \
      | grep -qx '.*/sync_loop\.sh' || continue
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$want" ] || continue
    echo "$p"
  done
}

cf404_sync_loops(){  # <local dir>
  cf404_sync_loop_pids "${1:?local dir}" | wc -l | tr -d ' '
}

# Stop the sync loop of ONE local root, by pid.
#
# NEVER `pkill` with a pattern here. On 2026-08-19 a pattern for this loop also
# matched four running eval shards, because an eval command line carries the
# sync root, and the four evals died. elisa is shared with other sessions, so a
# pattern can also reach a process this study does not own. The working
# directory identifies the loop, and the pid is what takes the signal.
cf404_stop_sync_loop(){  # <local dir>
  local p n=0
  for p in $(cf404_sync_loop_pids "${1:?local dir}"); do
    kill -TERM "$p" 2>/dev/null && n=$(( n + 1 ))
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

# How many cards this machine carries. `CF404_GPU_COUNT` overrides it, for a
# test that must not depend on the machine it runs on.
cf404_gpu_count(){
  if [ -n "${CF404_GPU_COUNT:-}" ]; then printf '%s\n' "$CF404_GPU_COUNT"; return 0; fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .
}

# Every index this machine carries, as `0 1 ... n-1`. This is what a launcher
# takes when the caller names no card, so a default can never name a card that
# is not there.
cf404_default_gpus(){
  local n i out=""
  n="$(cf404_gpu_count)"
  case "$n" in ''|*[!0-9]*) n=0 ;; esac
  for (( i = 0; i < n; i++ )); do out="$out $i"; done
  printf '%s\n' "${out# }"
}

# Refuse a card index this machine does not carry.
#
# On 2026-08-19 round 3's plan print put arm `a095` on `gpu=1` while the box
# held ONE card, at index 0. The print came from a dry run that passed no
# GPUS, so the launcher took its own default `0 1`. A lane on card 1 of a
# one-card box dies inside `.to(device)`, hours after the operator left.
#
# So the launchers ask this first, and they ask it on the machine that will
# hold the lane. It reads the real card count off the driver.
cf404_require_gpus(){  # <space separated indices>
  local n g bad=0
  n="$(cf404_gpu_count)"
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
    echo "  are 0 to $(( n - 1 )). A lane on a card that is not there dies" >&2
    echo "  inside .to(device) after the operator has left." >&2
    bad=1
  done
  [ "$bad" -eq 0 ]
}
