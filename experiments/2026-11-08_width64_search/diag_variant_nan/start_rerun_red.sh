#!/bin/bash
# The batch of the variant's step 13,755 (its CF_NAN_DEBUG dump) through red's model at one of red's saves, with full
# tensors (rerun_step.py with RERUN_WEIGHTS=resume): red's own command (the dry run of its run script) with --resume
# of the save given and a save folder of its own, no gradient checkpointing, on the CPU (fp32). It trains nothing and
# writes nothing in red's folder (red trains on GPU 1 meanwhile).
# Usage: bash start_rerun_red.sh <red save .pth> <tag> [check]
set -uo pipefail
D=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/diag_variant_nan
NAME=g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_sigf_align10pw2_rollfix
V=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/$NAME
WT=${WT:-${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/code_00060f42}
SAVE=$1 TAG=$2
[ -s "$SAVE" ] && [ -s "${SAVE%.pth}_optimizer.pth" ] || { echo "no save pair at $SAVE"; exit 2; }
LINE="$(cd "$V" && ABC_GIFT_DRY_RUN=1 BB_GPU=0 bash "run_$NAME.sh")" || exit 2
ENVS="${LINE%% python3 -u *}"
ENVS="${ENVS/FCST_GRAD_CKPT=1/FCST_GRAD_CKPT=0}"
ENVS="${ENVS/PATCH_ENC_CKPT=1/PATCH_ENC_CKPT=0}"
ARGS="${LINE#* python3 -u }"
TRAIN="${ARGS%% *}"
ARGS="${ARGS#* }"
OUT=$D/rerun_red_$TAG
ARGS="$(sed -E "s#--resume [^ ]+#--resume $SAVE#; s#--save-dir $V/ckpt #--save-dir $OUT/ckpt #" <<< "$ARGS")"
case "$ARGS" in *"--resume $SAVE "*"--save-dir $OUT/ckpt "*|*"--save-dir $OUT/ckpt "*"--resume $SAVE "*) ;;
  *) echo "--resume or --save-dir was not replaced"; exit 2;; esac
CMD="env $ENVS CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CF_NAN_DEBUG=1 RERUN_DUMP=$D/nan_dump.pt RERUN_OUT=$OUT/records.pt RERUN_MODE=train RERUN_SEED=0 RERUN_WEIGHTS=resume RERUN_STEP=13755 python3 -u $D/rerun_step.py $TRAIN $ARGS --device cpu"
if [ "${3:-}" = check ]; then echo "$CMD"; exit 0; fi
mkdir -p "$OUT/ckpt"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")" HUGGING_FACE_HUB_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
cd "$D" && exec $CMD
