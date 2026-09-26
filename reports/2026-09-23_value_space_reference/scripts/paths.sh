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
# The owner asked for the Moirai training recipe, with its schedule. The
# Moirai papers train at batch 256, AdamW at 1e-3, weight decay 0.1 and betas
# (0.9, 0.98), with a linear warmup over the first 10,000 steps and then a
# cosine anneal to 0 (Woo and others 2024, and Moirai 2.0, arXiv 2511.11698).
# The uni2ts pretraining config clips the gradient norm at 1.0
# (cli/conf/pretrain/default.yaml).
#
# The first run of this card used a flat 1e-3 with no warmup and no clip, as
# `experiments/2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/` did. Its
# loss spiked at 25k to 50k, 73k, 77k and 79k to 84k steps. The owner stopped
# it at 92,600 steps.
#
# One pass over the data is 166,000 steps at batch 256 (42.6M rows). The
# anneal ends there, and every leg names that length: unnamed, the trainer
# takes the leg's own --total-steps.
CF415_LR="${LR:-1e-3}"
CF415_LR_FINAL="${CF415_LR_FINAL:-0}"
CF415_LR_WARMUP_STEPS="${CF415_LR_WARMUP_STEPS:-10000}"
CF415_LR_COSINE_STEPS="${CF415_LR_COSINE_STEPS:-166000}"
CF415_GRAD_CLIP="${CF415_GRAD_CLIP:-1.0}"
CF415_BATCH_SIZE="${CF415_BATCH_SIZE:-256}"

# ---- The stops ---------------------------------------------------------------
#
# The eight the card scores. 166,000 steps at batch 256 is one pass.
CF415_STOPS="${CF415_STOPS:-10000 25000 50000 75000 100000 125000 150000 166000}"
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
