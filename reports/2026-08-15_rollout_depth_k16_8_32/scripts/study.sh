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
# It exists because phase 1 spends 19 hours of backbone time before the head
# half of this pipeline runs for the first time. A trial puts the first defect
# minutes from the start instead of hours.
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

# The checkpoint a stop produced, or nothing. The `*` tolerates train.py's
# `_rN` infix on a re-fired leg.
cf401_bb_ckpt(){  # <k> <stop steps>
  local dir name
  dir="$(cf401_leg_dir "${1:?k}" "${2:?stop}")"
  name="$(cf401_run_name "$1")"
  ls "$dir/$name"*_"$(( $2 / 1000 ))"k.pth 2>/dev/null \
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
