#!/bin/bash
# The replay of pass 2 of the variant 'red without SIGReg-F' with read-only probes (diag_replay.py): the run's own
# command (the dry run of its run script, resumed from its 10k save, as each restart of pass 2) with the save folder
# of this replay, on elisa GPU 0, under CF_NAN_DEBUG=1 (it stops at the first non-finite loss, gradient or weight
# and writes nan_dump.pt). Usage: bash start_replay.sh [check]  ('check' prints the command and runs nothing)
set -uo pipefail
D=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/diag_variant_nan
NAME=g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix
V=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/$NAME
WT=${WT:-${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/code_00060f42}
LINE="$(cd "$V" && ABC_GIFT_DRY_RUN=1 BB_GPU=0 bash "run_$NAME.sh")" || exit 2
case "$LINE" in *"--resume $V/ckpt/${NAME}_10k.pth "*) ;; *) echo "the dry run does not resume from the 10k save"; exit 2;; esac
ENVS="${LINE%% python3 -u *}"
ARGS="${LINE#* python3 -u }"
TRAIN="${ARGS%% *}"
ARGS="${ARGS#* }"
ARGS="${ARGS/--save-dir $V\/ckpt/--save-dir $D/ckpt}"
case "$ARGS" in *"--save-dir $D/ckpt "*) ;; *) echo "the save folder was not replaced"; exit 2;; esac
WINDOWS="10001-10030,10440-10620,11880-12060,13680-13760"
CMD="env $ENVS CF_NAN_DEBUG=1 CF_NAN_DEBUG_DUMP=$D/nan_dump.pt DIAG_OUT=$D DIAG_WINDOWS=$WINDOWS python3 -u $D/diag_replay.py $TRAIN $ARGS"
if [ "${1:-}" = check ]; then echo "$CMD"; exit 0; fi
mkdir -p "$D/ckpt" "$D/logs"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")" HUGGING_FACE_HUB_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
cd "$D" && exec $CMD
