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
# comparable, so nothing here may replace it.
set -uo pipefail

CF415_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF415_STUDY="$(cd "$CF415_SCRIPTS/.." && pwd)"
CF415_REPO="$(cd "$CF415_STUDY/../.." && pwd)"
# The checkout the trainer, the head trainer and the eval run from. Unset,
# this repository.
CF415_WT="${WT:-$CF415_REPO}"
# #373's directory. The leg runner, the head runner and the eval all live
# there, and this card reuses them rather than copying a second protocol that
# drifts.
CF415_PARENT="$CF415_WT/reports/2026-08-08_rollout_depth"
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
# `experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/` ran this
# recipe against this corpus at H = 384: lr 1e-3, weight decay 0.1, betas
# (0.9, 0.98), flat schedule, no warmup, no grad clip. It scored 1.183, the
# project best at the time.
#
# The BATCH SIZE is 64, not that run's 256. It has to be: 665,000 steps is one
# pass over the data at 64 rows a step, and #414's trajectory is measured on
# that clock. A twin at 256 would see four times the data at every stop.
CF415_LR="${LR:-1e-3}"
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
# The durable root. `leg_paths.sh` refuses /tmp and the checkout.
CF415_RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-415}"

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

cf415_is_stop(){  # <steps>
  local s
  for s in $CF415_STOPS; do [ "$s" = "${1:-}" ] && return 0; done
  echo "ABORT: stop='${1:-}' is not a stop of this card ($CF415_STOPS)" >&2
  return 2
}
