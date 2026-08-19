#!/bin/bash
# #401 — one configuration, two rollout depths, the MEAN objective. Sourced,
# never run.
#
# The configuration is #373's cell A4: `arm6_v2 combab`, `L_align` on the
# student, scheduled EMA. It set the project's best GM-Relative MASE, 1.0660.
# #373 already holds it as the `arm6_v2_combab_alignS` case of
# `run_leg_k.sh`, so this study adds no trainer call. It supplies four
# things: the depth, the reduction over the depth copies, the stop, and the
# head budget.
#
# Everything here is a constant or a path. The scripts that act on them are
# run_arm_k.sh (the backbone), head_eval.sh (the head and the 97-config
# GIFT-Eval), phase1.sh and phase2.sh (the order), launch_box.sh and
# launch_elisa.sh (which machine does what).
#
# ---- The protocol, and the arm it is compared against -------------------------
#
# The first run of this card summed the k + 1 depth copies, which is what
# `--train-rollout-depth` did before #401. The f-side then carried k + 1
# times its k = 0 weight against the terms that hold no f. Every k > 0
# backbone of that run took ONE DIRECTION for every series and kept a scalar
# readout, at every depth and every stop, and every cell scored well above
# the k = 0 parent.
#
# That run is stopped. Its 8 scored cells stay in `results/` as the
# comparison arm. This protocol takes the MEAN over the k + 1 copies
# (`--train-rollout-reduce mean`, docs/train_rollout_depth.md), so the f-side
# holds its k = 0 weight at every depth, and the depth changes what the model
# trains on rather than how much the f-side outweighs the rest.
#
# The two arms never share a file. The reduction picks the run name, the
# checkpoint root and the results directory — see CF401_REDUCE below.

# The card's cell, its depths and its stops. Two depths, not three: k = 8 and
# k = 32 bracket the range, and dropping k = 16 buys the answer sooner.
CF401_CELL="arm6_v2_combab_alignS"
CF401_DEPTHS="8 32"
CF401_STOPS="40000 100000 200000"
CF401_ENC="student"

# How the k + 1 depth copies combine. `mean` is this protocol. `sum`
# reproduces the stopped arm and writes where that arm wrote.
CF401_REDUCE="${CF401_REDUCE:-mean}"
case "$CF401_REDUCE" in
  sum|mean) ;;
  *) echo "ABORT: CF401_REDUCE='$CF401_REDUCE' is not sum or mean" >&2
     return 2 2>/dev/null || exit 2 ;;
esac

# Phase 1 trains one head per backbone stop, at a fixed budget. Phase 2
# repeats the head with its budget matched to the backbone stop, which is
# the second half of the card's question. Phase 2 therefore has no constant:
# its budget IS the stop, and phase2.sh and cf401_require_head_steps both
# read it off the stop they are given.
CF401_HEAD_STEPS_P1=30000

# ---- Trial mode --------------------------------------------------------------
# `CF401_TRIAL=<backbone steps>` runs the WHOLE pipeline at a budget that
# finishes in minutes. It replaces two step counts and two paths. It replaces
# no script, no flag and no guard: the same run_arm_k.sh, the same
# head_eval.sh, the same #373 runner, head trainer and eval, and the same
# refusals over the replaced lists.
#
# It exists because phase 1 trains the k = 8 arm to its first stop before the
# head half of this pipeline runs at all: 40,000 steps at 288.4 ms is 3.2 hours
# (results/mean/leg_cost.csv). A trial puts the first defect minutes from the
# start instead of hours.
#
#   CF401_TRIAL=400 bash scripts/trial_head.sh
#
# The trial's one stop is also its phase-2 head budget, so the card's phase-2
# rule (head steps = backbone steps) holds unchanged. Its phase-1 budget is
# half of that, so the two phases still write two tags.
if [ -n "${CF401_TRIAL:-}" ]; then
  CF401_STOPS="$CF401_TRIAL"
  CF401_HEAD_STEPS_P1=$(( CF401_TRIAL / 2 ))
fi

CF401_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF401_STUDY="$(dirname "$CF401_SCRIPTS")"

# The durable root. Never /tmp, never inside the checkout (CLAUDE.md
# checkpoint safety rule 4), and never #373's root — one root for two studies
# is a sync loop that cannot tell their checkpoints apart.
CF401_ROOT_DEFAULT="/home/jupyter/checkpoints_backup/cf-401"

# ---- Which machine owns which arm --------------------------------------------
#
# The BOX owns both backbone arms, k = 8 and k = 32, one GPU each. Its root is
# the single source for backbones. elisa owns every head and every 97-config
# GIFT-Eval, on GPU 0. elisa GPU 1 belongs to another session.
#
# So the two machines read two different roots, and the two are ONE tree: the
# box saves to CF401_BOX_RUNS on its own disk, the sync loop pulls that tree
# into CF401_SYNC_DIR keeping the relative paths, and CF401_SYNC_ROOT is what
# elisa reads. Each launcher takes its own role's root through
# `cf401_use_root`.
#
# Two roots that are NOT one tree is the failure this layout removes: a
# `heads_watch.sh` reads one CF401_ROOT, so a second arm elsewhere climbs for
# 33 h and is never scored. `cf401_foreign_arm` refuses that case out loud.
CF401_BOX_LABEL="${CF401_BOX_LABEL:-box_a}"
CF401_BOX_RUNS="${CF401_BOX_RUNS:-/root/cf401_runs}"
CF401_SYNC_DIR="${CF401_SYNC_DIR:-$HOME/cf401_sync/$CF401_BOX_LABEL}"
CF401_SYNC_ROOT="${CF401_SYNC_ROOT:-$CF401_SYNC_DIR/sync}"

# The mean arm writes nowhere the summed arm wrote. Four names carry the
# reduction: the checkpoint root, the results directory, the plots directory
# and the run name (see cf401_run_name). Without all four, this protocol's
# `score_k8_bb40k_h30k_student.txt` and its figures would overwrite the
# stopped arm's, which is the comparison.
#
# The reduction picks the DEFAULTS. A path given on the command line is taken
# as it is, for two reasons.
#
#   Neither machine can rename the path it hands in. The box saves to
#   /root/cf401_runs, and on elisa the root IS where the sync loop lands the
#   box's tree, `<LOCAL_DIR>/sync`. A suffix on top of either would point the
#   heads at a directory that holds nothing.
#
#   A launcher sets these on its command line, which EXPORTS them, and every
#   script it spawns sources this file again. A suffix applied on top of an
#   inherited value would stack: `cf-401-mean`, then `cf-401-mean-mean` one
#   level down, and each level would read a different tree.
#
# The run name carries the reduction either way, so two objectives under one
# overridden root still write two checkpoint sets.
# Whether the caller PINNED the root, captured before the defaults below take
# it. A pinned root is a deliberate "read this tree and no other", and
# `cf401_eval_csv` must not widen its search past it — a test that points the
# root at an empty directory, to check the no-eval path, would otherwise find
# the real study's evals and read another run's numbers as its own.
#
# A launcher that exports the root counts as pinned, and that is right: it
# pinned it through `cf401_use_root`, which already resolved to the tree it
# wants.
if [ -n "${CF401_ROOT:-}" ]; then CF401_ROOT_PINNED=1
else CF401_ROOT_PINNED=0; fi

if [ "$CF401_REDUCE" = "sum" ]; then
  CF401_RUN_SUFFIX=""
  CF401_ROOT="${CF401_ROOT:-$CF401_ROOT_DEFAULT}"
  CF401_RESULTS="${CF401_RESULTS:-$CF401_STUDY/results}"
  CF401_PLOTS="${CF401_PLOTS:-$CF401_STUDY/plots}"
else
  CF401_RUN_SUFFIX="_$CF401_REDUCE"
  CF401_ROOT="${CF401_ROOT:-$CF401_ROOT_DEFAULT-$CF401_REDUCE}"
  CF401_RESULTS="${CF401_RESULTS:-$CF401_STUDY/results/$CF401_REDUCE}"
  CF401_PLOTS="${CF401_PLOTS:-$CF401_STUDY/plots/$CF401_REDUCE}"
fi

# A trial writes nowhere the study writes. This suffix IS applied on top of
# an override, because a trial that wrote into the real root would put
# `leg_400/` beside the study's stops and a throwaway checkpoint under a name
# a report reads. It is applied once: a launcher exports the suffixed value
# and its children source this file again.
if [ -n "${CF401_TRIAL:-}" ]; then
  case "${CF401_ROOT%/}" in
    *-trial) CF401_ROOT="${CF401_ROOT%/}" ;;
    *) CF401_ROOT="${CF401_ROOT%/}-trial" ;;
  esac
  case "${CF401_RESULTS%/}" in
    */trial) CF401_RESULTS="${CF401_RESULTS%/}" ;;
    *) CF401_RESULTS="${CF401_RESULTS%/}/trial" ;;
  esac
fi
# #373's directory, which holds the runner, the head trainer and the eval
# this study reuses. Resolved from this file, so a checkout at any path works.
CF401_PARENT="$(cd "$CF401_STUDY/../2026-08-08_rollout_depth" && pwd)"
CF401_REPO="$(cd "$CF401_STUDY/../.." && pwd)"

# The checkout the trainer and the head trainer come from. `run_leg_k.sh`
# reads `$WT/experiments/...`, so it must be a checkout, not a results dir.
CF401_WT="${WT:-$CF401_REPO}"

# The run name `run_leg_k.sh` builds for this cell at this depth, through its
# RUN_SUFFIX. It carries the depth, so no name collides with a published #373
# one, and it carries the reduction, so the mean arm's checkpoints and losses
# CSV cannot be read as the summed arm's.
cf401_run_name(){  # <k>
  printf 'cf393_%s_cf373k%s%s\n' "$CF401_CELL" "${1:?k}" "$CF401_RUN_SUFFIX"
}

# The root ONE arm saves under. The arms are one cell at two depths, so
# `run_leg_k.sh` would lay both into one <root>/<cell>/leg_<N>k/. The run
# names carry the depth and do not collide, but a save dir shared by two runs
# is CLAUDE.md checkpoint safety rule 3, and a glob written later that
# forgets the depth would resolve to the wrong arm's checkpoint. One root per
# depth costs nothing and removes the question.
cf401_arm_root(){  # <k>
  printf '%s/k%s\n' "$CF401_ROOT" "${1:?k}"
}

# Where a leg saves. `run_leg_k.sh` lays out <root>/<cell>/leg_<N>k/.
cf401_leg_dir(){  # <k> <stop steps>
  printf '%s/%s/leg_%dk\n' "$(cf401_arm_root "${1:?k}")" "$CF401_CELL" \
    "$(( ${2:?stop} / 1000 ))"
}

# The checkpoint a stop produced, or nothing.
#
# Two names, not one glob. `<name>_<N>k.pth` is the leg's own, and
# `<name>_r<N>_<N>k.pth` is train.py's `_rN` infix on a re-fired leg
# (`safe_run_name`). A trailing `*` would take BOTH names, and one more: the
# summed arm's name is a prefix of the mean arm's, so `sum`'s glob would
# resolve to `..._mean_40k.pth` and score the other objective's backbone.
cf401_bb_ckpt(){  # <k> <stop steps>
  local dir name kk
  dir="$(cf401_leg_dir "${1:?k}" "${2:?stop}")"
  name="$(cf401_run_name "$1")"
  kk=$(( $2 / 1000 ))
  ls "$dir/$name"_"$kk"k.pth "$dir/$name"_r[0-9]*_"$kk"k.pth 2>/dev/null \
    | grep -v optimizer | head -1
}

# A step count, as a tag reads it. `40000` -> `40k`, `400` -> `400`.
#
# The `k` form is the study's own, and every published number carries it. A
# trial budget is not a multiple of 1000, and two trial budgets both rounded
# to `0k` would give the two phases ONE tag — so phase 2 would find phase 1's
# score file already written and skip.
cf401_steps_label(){  # <steps>
  local n="${1:?steps}"
  if [ $(( n % 1000 )) -eq 0 ]; then printf '%dk' $(( n / 1000 ))
  else printf '%d' "$n"; fi
}

# The inverse, for the reader in collect.sh.
cf401_steps_of(){  # <label>
  case "${1:?label}" in
    *k) printf '%d' $(( ${1%k} * 1000 )) ;;
    *)  printf '%d' "${1}" ;;
  esac
}

# The name of one (depth, stop, head budget). It names the head checkpoint,
# the eval directory and the score file, so the two phases never share one.
cf401_tag(){  # <k> <stop steps> <head steps>
  printf 'k%s_bb%s_h%s_%s\n' "${1:?k}" \
    "$(cf401_steps_label "${2:?stop}")" \
    "$(cf401_steps_label "${3:?head steps}")" "$CF401_ENC"
}

# Where #373's head script lays one tag's head, its GIFT-Eval output and its
# merged 97-config CSV. One root per depth, the same rule as the backbones —
# see cf401_arm_root.
cf401_eval_dir(){  # <k> <tag>
  printf '%s/eval/%s\n' "$(cf401_arm_root "${1:?k}")" "${2:?tag}"
}

# The eval's own 97-config CSV for one tag, wherever that tag's eval landed.
#
# A tag's eval is under one of a few roots, in one of two layouts, and which
# one depends on who ran the head rather than on anything in the tag.
#
#   roots    CF401_ROOT is whatever the caller holds. `heads_watch.sh` points
#            it at the sync tree the box lands in, through `cf401_use_root`,
#            and a plain `collect.sh` keeps the study default. Neither knows
#            the other's, so both are tried — but the sync tree ONLY when the
#            caller left the root at its default. A pinned root means read
#            that tree and no other. A VARIANT cell is a side run under its
#            own root, `<study root>-<variant>`, which is tried against both
#            of those bases for the same reason.
#   layouts  this study lays an eval at <root>/k<K>/eval/<tag>. #373's runner,
#            which the variant cell was driven by hand through, lays it one
#            level deeper, at <root>/k<K>/<cell>/eval/<tag>.
#
# A tag names exactly one eval, so no order over these candidates can return
# another cell's numbers. When none exists, this study's own path under
# CF401_ROOT is printed unchanged, and `split_scores.py` reports the miss as
# it did before.
cf401_eval_csv(){  # <k> <tag> [variant]
  local k="${1:?k}" tag="${2:?tag}" var="${3:-base}" first root c roots
  first="$(cf401_eval_dir "$k" "$tag")/gift/all_results.csv"
  roots=("${CF401_ROOT%/}" "${CF401_ROOT%/}-$var")
  [ "${CF401_ROOT_PINNED:-0}" -eq 1 ] || roots+=(
    "${CF401_SYNC_ROOT%/}" "$CF401_ROOT_DEFAULT-$CF401_REDUCE-$var")
  for root in "${roots[@]}"; do
    [ -n "$root" ] || continue
    for c in "$root/k$k/eval/$tag/gift/all_results.csv" \
             "$root/k$k/$CF401_CELL/eval/$tag/gift/all_results.csv"; do
      [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
  done
  printf '%s\n' "$first"
}

cf401_is_in(){  # <value> <space separated list>
  case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# How many `cos_err_dj` columns a run's losses CSV carries. A k-depth run
# writes k + 1 of them (docs/train_rollout_depth.md), so this is the proof
# that the flag reached the depth it was given. Prints nothing when no CSV is
# under the root.
#
# `tr -d '\r'` is not defensive: the trainer's CSV writer ends every line
# CRLF, so the LAST field of the header carries a trailing \r and an anchored
# match misses it. Without it the count reads k, which is off by one and
# still plausible. That is why this lives here, in a file a test can source,
# and not inside the smoke script.
cf401_depth_cols(){  # <runs root>
  local csv
  csv="$(ls "${1:?root}"/*/leg_*k/*_losses.csv 2>/dev/null | head -1)"
  [ -n "$csv" ] || { echo ""; return; }
  # `grep -c` prints 0 and exits 1 when it matches nothing, which is the
  # normal k = 0 case. The count is the answer, so the status is dropped.
  head -1 "$csv" | tr -d '\r' | tr ',' '\n' | grep -c '^cos_err_d[0-9]*$' || true
}

# ---- What the trainer of a leg actually runs ----------------------------------
#
# The depth has a proof in the artefacts: `cf401_depth_cols` counts the
# `cos_err_dj` columns. The REDUCTION has none. A mean leg that trained the sum
# writes the same file names, the same columns and the same log lines.
#
# The trainer's own command line is the one place that names it. Nothing else
# does: the reduction rides in through `GAP_ARGS` and the root through
# `--save-dir`, so neither is in a wrapper's argv. train.py writes that command
# line as the first line of every run's log, and the functions below read it
# from the log (this leg, and every earlier one) or from /proc (a leg that runs
# now, whoever started it).

# The reduction a trainer command line names. `sum` when it carries no flag,
# because `sum` is train.py's own default — so a line with no flag trains it.
# Reads the command line on stdin, NUL-separated (a /proc cmdline) or
# space-separated (a log line). Both `--flag value` and `--flag=value` forms.
cf401_reduce_of_cmdline(){
  tr '\0 ' '\n\n' | awk -F= '
    $1 == "--train-rollout-reduce" {
      if (NF > 1) { print $2 } else { getline; print }
      found = 1; exit }
    END { if (!found) print "sum" }'
}

# The runs root a leg's `--save-dir` implies, or nothing when the path is not
# THIS study's layout. `run_leg_k.sh` saves to <root>/k<K>/<cell>/leg_<N>k, so
# three levels come off. #373's own legs save one level shallower and do not
# match, which is what keeps the guards below off another study's runs.
cf401_root_of_save_dir(){  # <save dir>
  local d="${1:-}" leg cell arm
  d="${d%/}"
  leg="${d##*/}";  d="${d%/*}"
  cell="${d##*/}"; d="${d%/*}"
  arm="${d##*/}";  d="${d%/*}"
  case "$leg" in leg_[0-9]*k) ;; *) return 1 ;; esac
  [ "$cell" = "$CF401_CELL" ] || return 1
  case "$arm" in k[0-9]*) ;; *) return 1 ;; esac
  [ -n "$d" ] || return 1
  printf '%s\n' "$d"
}

# Every leg of this study's layout that runs on THIS machine, as
# `<pid> <reduction> <runs root>`.
cf401_running_legs(){
  local pid cmd save red root
  for pid in $(pgrep -f 'train\.py' 2>/dev/null); do
    [ -r "/proc/$pid/cmdline" ] || continue
    cmd="$(tr '\0' '\n' <"/proc/$pid/cmdline" 2>/dev/null)"
    save="$(printf '%s\n' "$cmd" | awk -F= '
      $1 == "--save-dir" { if (NF > 1) { print $2 } else { getline; print }
                           exit }')"
    [ -n "$save" ] || continue
    root="$(cf401_root_of_save_dir "$save")" || continue
    red="$(printf '%s\n' "$cmd" | cf401_reduce_of_cmdline)"
    printf '%s %s %s\n' "$pid" "$red" "$root"
  done
}

# Every leg that trains THIS reduction under ANOTHER root. Prints one line per
# leg and returns 0 when it found any, so a caller can both refuse and name
# what it refused. Reads `<pid> <reduction> <root>` lines on stdin, so the
# caller decides where the list comes from.
#
# A leg of the other reduction is not this arm's business: the two arms share
# no file. A leg under the same root is the normal case — on the box the arms
# and the root are one tree, and on elisa the root is that tree as the sync
# loop lands it.
cf401_foreign_arm(){  # <reduction> <root>
  local want="${1:?reduction}" mine="${2:?root}" list pid red root hits=0
  mine="${mine%/}"
  # The list is read whole first. `read` needs a final newline, and a caller
  # that pipes one line without one would otherwise look like an empty list —
  # which is the answer "nothing is running", the opposite of what it holds.
  list="$(cat)"
  [ -n "$list" ] || return 1
  while read -r pid red root; do
    [ -n "$pid" ] || continue
    [ "$red" = "$want" ] || continue
    [ "${root%/}" = "$mine" ] && continue
    printf '%s %s %s\n' "$pid" "$red" "$root"
    hits=$(( hits + 1 ))
  done <<<"$list"
  [ "$hits" -gt 0 ]
}

# The root a machine's ROLE implies: CF401_BOX_RUNS on the box, which owns both
# backbone arms, and CF401_SYNC_ROOT on elisa, which reads that tree.
#
# Called by a launcher that captured `CF401_ROOT_GIVEN` BEFORE it sourced this
# file, so an operator's own root still wins and this default is not mistaken
# for one. Exported, because the launcher's children source this file again.
cf401_use_root(){  # <root>
  CF401_ROOT="${CF401_ROOT_GIVEN:-${1:?root}}"
  export CF401_ROOT
}

# The log #373's runner writes a leg's trainer output to.
cf401_leg_log(){  # <k>
  printf '%s/run_%s.log\n' "$CF401_RESULTS" "$(cf401_run_name "${1:?k}")"
}

# How many command lines a leg log holds. The runner APPENDS, so a resumed
# cell's log carries one per leg. A caller that counts before it starts a leg
# knows when THIS leg's line has landed.
cf401_cmdlines(){  # <trainer log>
  grep -c '^Command line:' "${1:?log}" 2>/dev/null || true
}

# The reduction the trainer of this leg runs, out of the LAST command line in
# its log. Prints nothing, and returns non-zero, when the log holds none.
cf401_leg_reduce(){  # <trainer log>
  local line
  line="$(grep '^Command line:' "${1:?log}" 2>/dev/null | tail -1)"
  [ -n "$line" ] || return 1
  printf '%s' "${line#Command line: }" | cf401_reduce_of_cmdline
}

# Every process under a tree, leaves first, so a signal reaches the trainer
# before the wrapper that would otherwise report its death first.
cf401_kill_tree(){  # <pid>
  local kid
  for kid in $(pgrep -P "${1:?pid}" 2>/dev/null); do cf401_kill_tree "$kid"; done
  kill -TERM "$1" 2>/dev/null
}

# How many sync loops run for ONE local root, identified by their working
# directory. elisa is shared: another study's loop is not this study's, and a
# count over every `sync_loop.sh` on the machine reports a pull that is not
# happening.
#
# `pgrep -f` also matches a process that merely NAMES the file on its command
# line, this check among them, so the argument list decides and not the
# pattern. `wc -l`, because `pgrep -c` prints 0 AND exits 1 on no match.
cf401_sync_loops(){  # <local dir>
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

# Every guard prints what it refused. A depth or a stop that is not the
# card's is a typo in a launch command, and a typo that trains for six hours
# is expensive.
cf401_require_depth(){  # <k>
  cf401_is_in "${1:-}" "$CF401_DEPTHS" && return 0
  echo "ABORT: k='${1:-}' is not a depth of this study ($CF401_DEPTHS)" >&2
  return 2
}

cf401_require_stop(){  # <stop steps>
  cf401_is_in "${1:-}" "$CF401_STOPS" && return 0
  echo "ABORT: stop='${1:-}' is not a stop of this study ($CF401_STOPS)" >&2
  return 2
}

# The card defines exactly two head budgets for a stop: the phase-1 fixed
# 30,000, and the phase-2 one, which IS the backbone step count. A list test
# over all four numbers accepted `head_eval.sh 8 100000 40000` — 40,000 is a
# phase-2 budget, but not this stop's. That writes a tag neither phase
# defines, and collect.sh then reads it as phase 1. So the phase-2 arm is
# tested against THIS stop, not against the list.
cf401_require_head_steps(){  # <head steps> <stop steps>
  local h="${1:-}" s="${2:-}"
  [ "$h" = "$CF401_HEAD_STEPS_P1" ] && return 0
  [ -n "$s" ] && [ "$h" = "$s" ] && return 0
  echo "ABORT: head steps='$h' is neither the phase-1 budget" >&2
  echo "  ($CF401_HEAD_STEPS_P1) nor the phase-2 budget for stop='$s'" >&2
  return 2
}
