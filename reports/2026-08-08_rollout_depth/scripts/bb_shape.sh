#!/bin/bash
# #373 — the backbone shape the head trainer and the GIFT-Eval build to.
#
# Both scripts rebuild the backbone before they load its weights, and both
# take the shape from the command line. `head_eval_bb.sh` and `eval_local.sh`
# held the same three flags as two literals, so a card that trains another
# shape had to find and change both.
#
# #412 trains this cell at `d_model` 384, about 11.4 million parameters. A
# head that builds a `d_model` 64 backbone cannot load it, and the run stops
# after it takes the card.
#
# Unset, `BB_SHAPE` is the shape every published run of #373, #393, #401, #404
# and #409 used, so every command line of those cards reproduces.
#
# `head_eval_bb.sh` starts `eval_local.sh` as a child, so ONE exported
# `CF_BB_SHAPE` reaches the head and the evaluation together.
#
#   CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" bash head_eval_bb.sh ...
#
# The encoder stack is NOT here. Both scripts read `num_encoder_layers` off
# the `transformer.encoder_layers.<N>.*` keys of the checkpoint, so a flag
# would be a second source for one number.
read -r -a BB_SHAPE <<<"${CF_BB_SHAPE:---d-model 64 --n-heads 8 --num-layers 3}"
