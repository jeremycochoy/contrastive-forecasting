#!/bin/bash
# #401 — one configuration, three rollout depths. Sourced, never run.
#
# The configuration is #373's cell A4: `arm6_v2 combab`, `L_align` on the
# student, scheduled EMA. It set the project's best GM-Relative MASE, 1.0660.
# #373 already holds it as the `arm6_v2_combab_alignS` case of
# `run_leg_k.sh`, so this study adds no flag and no trainer call. It supplies
# three numbers: the depth, the stop, and the head budget.
#
# Everything here is a constant or a path. The scripts that act on them are
# run_arm_k.sh (the backbone), head_eval.sh (the head and the 97-config
# GIFT-Eval), phase1.sh and phase2.sh (the order).

# The card's cell, its depths and its stops. The depth order is the card's:
# k = 16 answers the question first, k = 8 and k = 32 bracket it.
CF401_CELL="arm6_v2_combab_alignS"
CF401_DEPTHS="16 8 32"
CF401_STOPS="40000 100000 200000"
CF401_ENC="student"

# Phase 1 trains one head per backbone stop, at a fixed budget. Phase 2
# repeats the head with its budget matched to the backbone stop, which is
# the second half of the card's question.
CF401_HEAD_STEPS_P1=30000
CF401_HEAD_STEPS_P2="$CF401_STOPS"

# The durable root. Never /tmp, never inside the checkout (CLAUDE.md
# checkpoint safety rule 4), and never #373's root — one root for two studies
# is a sync loop that cannot tell their checkpoints apart.
CF401_ROOT="${CF401_ROOT:-/home/jupyter/checkpoints_backup/cf-401}"

CF401_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF401_STUDY="$(dirname "$CF401_SCRIPTS")"
CF401_RESULTS="${CF401_RESULTS:-$CF401_STUDY/results}"
# #373's directory, which holds the runner, the head trainer and the eval
# this study reuses. Resolved from this file, so a checkout at any path works.
CF401_PARENT="$(cd "$CF401_STUDY/../2026-08-08_rollout_depth" && pwd)"
CF401_REPO="$(cd "$CF401_STUDY/../.." && pwd)"

# The checkout the trainer and the head trainer come from. `run_leg_k.sh`
# reads `$WT/experiments/...`, so it must be a checkout, not a results dir.
CF401_WT="${WT:-$CF401_REPO}"

# The run name `run_leg_k.sh` builds for this cell at this depth. It carries
# the depth, so no name of this study can collide with a published #373 one.
cf401_run_name(){  # <k>
  printf 'cf393_%s_cf373k%s\n' "$CF401_CELL" "${1:?k}"
}

# The root ONE arm saves under. The three arms are one cell at three depths,
# so `run_leg_k.sh` would lay all three into one <root>/<cell>/leg_<N>k/. The
# run names carry the depth and do not collide, but a save dir shared by three
# runs is CLAUDE.md checkpoint safety rule 3, and a glob written later that
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

# The name of one (depth, stop, head budget). It names the head checkpoint,
# the eval directory and the score file, so the two phases never share one.
cf401_tag(){  # <k> <stop steps> <head steps>
  printf 'k%s_bb%dk_h%dk_%s\n' "${1:?k}" "$(( ${2:?stop} / 1000 ))" \
    "$(( ${3:?head steps} / 1000 ))" "$CF401_ENC"
}

cf401_is_in(){  # <value> <space separated list>
  case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac
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

cf401_require_head_steps(){  # <head steps>
  cf401_is_in "${1:-}" "$CF401_HEAD_STEPS_P1 $CF401_HEAD_STEPS_P2" && return 0
  echo "ABORT: head steps='${1:-}' is neither the phase-1 budget" >&2
  echo "  ($CF401_HEAD_STEPS_P1) nor a phase-2 one ($CF401_HEAD_STEPS_P2)" >&2
  return 2
}
