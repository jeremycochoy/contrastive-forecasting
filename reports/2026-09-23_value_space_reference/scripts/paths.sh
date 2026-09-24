#!/bin/bash
# #415 — the paths, the shape and the stops of the value-space reference.
# Sourced, never run.
#
# The card trains ONE model: #414's cell `arm6_v2_combab_alignT` at `d_model`
# 384, with the objective replaced. So this file holds no arm table. Every
# value below is either the cell's, or the Moirai recipe the issue names.
#
# The scoring path is #414's, unchanged: #373's `head_eval_bb.sh` trains a
# quantile head on the frozen backbone and runs the 97-config GIFT-Eval under
# strategy B4 at forecast length 16. That is what makes the two numbers
# comparable, so nothing here may replace it. A second eval scores the same
# head under A2, the value-space rollout this model trains.
set -uo pipefail

CF415_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF415_STUDY="$(cd "$CF415_SCRIPTS/.." && pwd)"
CF415_REPO="$(cd "$CF415_STUDY/../.." && pwd)"
# The checkout the trainer, the head trainer and the eval run from. Unset,
# this repository.
CF415_WT="${WT:-$CF415_REPO}"
# #373's directory. The leg helpers, the head runner and the eval all live
# there, and this card reuses them rather than copying a second protocol that
# drifts. They come from THIS checkout, as #412's do: the A2 eval needs the
# `eval_local.sh` of the same commit as this card.
CF415_PARENT="$CF415_REPO/reports/2026-08-08_rollout_depth"
CF415_RESULTS="${CF_RESULTS:-$CF415_STUDY/results}"

# ---- The model ---------------------------------------------------------------
#
# #412 measured these three: `d_model` 384 over 3 encoder layers and 3
# forecaster layers is 11,431,548 trainable parameters, 0.3% over
# Moirai-2-Small. The value head adds 384 x 9 x 16 = 55,296 more, and it is a
# training-only branch that the head trainer and the eval strip.
CF415_D_MODEL=384
CF415_N_HEADS=8
CF415_NUM_LAYERS=3
CF415_NUM_ENCODER_LAYERS=3
# The rollout depth, in VALUE space. #414's cell trains k = 3 in latent space.
CF415_K="${K:-3}"
# The backbone seed of #414's best arm, so the two runs start from the same
# draw.
CF415_SEED="${SEED:-20260520}"

# ---- The Moirai recipe -------------------------------------------------------
#
# `experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/` ran the
# recipe against this corpus at H = 384. `scripts/run_resume50k.sh` line 44
# sets batch 256, and line 45 sets lr 1e-3, weight decay 0.1 and betas
# (0.9, 0.98). The report (line 94) and `scripts/run.sh` (lines 91 to 93)
# give no warmup, no grad clip and a flat rate.
#
# The BATCH SIZE stays at 64. 665,000 steps is one pass over the data at 64
# rows a step, and #414's trajectory is measured on that clock. Also, the
# box's card is shared, and a leg at 256 holds four times the activations.
#
# So the RATE moves with the batch: 1e-3 x sqrt(64 / 256) = 5e-4, the
# square-root rule for Adam. The weight decay and the betas stay.
#
# The SCHEDULE is not flat. #414 found that every constant rate reaches its
# best score at one step and then climbs, and its best arm, `cos200k`, anneals.
# The Moirai paper also anneals by cosine. So the rate falls by one cosine
# from 5e-4 to 1e-6 over one pass, 665,000 steps. Every leg names that length:
# unnamed, the trainer takes the leg's own --total-steps, and the first leg
# would anneal to the floor by step 40,000. The floor is #414's.
#
# NO WARMUP and NO GRAD CLIP, as in that run. The trainer has neither.
CF415_LR="${LR:-5e-4}"
CF415_LR_FINAL="${CF415_LR_FINAL:-1e-6}"
CF415_LR_COSINE_STEPS="${CF415_LR_COSINE_STEPS:-665000}"
CF415_BATCH_SIZE="${CF415_BATCH_SIZE:-64}"

# ---- The stops ---------------------------------------------------------------
#
# The eight the card scores. 665,000 steps is one pass over the data.
CF415_STOPS="${CF415_STOPS:-40000 100000 200000 300000 400000 500000 600000 665000}"
# #373's head protocol, which #412 and #414 run on every stop.
CF415_HEAD_STEPS="${CF415_HEAD_STEPS:-30000}"
# No teacher exists on this card, so the head reads the student encoder.
CF415_ENC="student"
CF415_HEAD_VRAM_MIB="${CF415_HEAD_VRAM_MIB:-9000}"

# The run name. It carries the card, so no checkpoint of this study reads as
# #373's, #412's or #414's.
CF415_RUN_NAME="${CF415_RUN_NAME:-cf415_value_k${CF415_K}}"
# The durable root, on the vast box that trains the leg. The orchestrator
# mirrors that disk to elisa, so the leg itself syncs nothing. `leg_paths.sh`
# refuses /tmp and the checkout. A machine without the box's disk names its
# own root, for example to score a mirrored stop on elisa.
CF415_RUNS="${RUNS:-/workspace/ckpt/cf-415}"

cf415_bb_shape(){
  printf -- '--d-model %s --n-heads %s --num-layers %s\n' \
    "$CF415_D_MODEL" "$CF415_N_HEADS" "$CF415_NUM_LAYERS"
}

cf415_steps_label(){  # <steps>
  printf '%dk\n' "$(( ${1:?steps} / 1000 ))"
}

cf415_tag(){  # <stop steps>
  printf 'value_bb%s_h%s_%s\n' "$(cf415_steps_label "${1:?stop}")" \
    "$(cf415_steps_label "$CF415_HEAD_STEPS")" "$CF415_ENC"
}

# One score file per stop and strategy. B4 keeps #414's name, so the two
# cards' files read alike.
cf415_score_file(){  # <stop steps> <B4|A2>
  case "${2:?strategy}" in
    B4) printf '%s/score_%s.txt\n' "$CF415_RESULTS" "$(cf415_tag "$1")" ;;
    A2) printf '%s/score_%s_a2.txt\n' "$CF415_RESULTS" "$(cf415_tag "$1")" ;;
    *) echo "ABORT: strategy '$2' is not B4 or A2" >&2; return 2 ;;
  esac
}

# The backbone checkpoint of one stop, or nothing. Needs `leg_paths.sh`.
cf415_bb_ckpt(){  # <stop steps>
  local root
  root="$(RUNS="$CF415_RUNS" runs_root)" || return 2
  ckpt_at_step "$root/value_space" "$CF415_RUN_NAME" "$(( ${1:?stop} / 1000 ))"
}

cf415_is_stop(){  # <steps>
  local s
  for s in $CF415_STOPS; do [ "$s" = "${1:-}" ] && return 0; done
  echo "ABORT: stop='${1:-}' is not a stop of this card ($CF415_STOPS)" >&2
  return 2
}
