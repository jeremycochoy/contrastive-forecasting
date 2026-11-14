#!/bin/bash
# One step of 'red without SIGReg-F' again from its CF_NAN_DEBUG dump, with full tensors (rerun_step.py): the run's
# own command (the dry run of its run script, as in start_replay.sh) with a save folder of its own and no gradient
# checkpointing (one record per forecaster call), on the CPU (fp32: the fp16 autocast of the layers is CUDA only)
# or on elisa GPU 0 (the run's fp16). It trains nothing.
# Usage: [RERUN_WEIGHTS=dump|resume] bash start_rerun.sh cpu|gpu train|eval <seed> [check]   ('check' prints the command and runs nothing)
set -uo pipefail
D=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/diag_variant_nan
NAME=g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10pw2_rollfix
V=${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/$NAME
WT=${WT:-${HP64_ROOT:-/home/jupyter/cf_runs/hp64}/code_00060f42}
DEV=$1 MODE=$2 SEED=$3 WEIGHTS=${RERUN_WEIGHTS:-dump}
LINE="$(cd "$V" && ABC_GIFT_DRY_RUN=1 BB_GPU=0 bash "run_$NAME.sh")" || exit 2
case "$LINE" in *"--resume $V/ckpt/${NAME}_10k.pth "*) ;; *) echo "the dry run does not resume from the 10k save"; exit 2;; esac
ENVS="${LINE%% python3 -u *}"
ENVS="${ENVS/FCST_GRAD_CKPT=1/FCST_GRAD_CKPT=0}"
ENVS="${ENVS/PATCH_ENC_CKPT=1/PATCH_ENC_CKPT=0}"
ARGS="${LINE#* python3 -u }"
TRAIN="${ARGS%% *}"
ARGS="${ARGS#* }"
OUT=$D/rerun_${DEV}_${MODE}_s${SEED}_${WEIGHTS}
ARGS="${ARGS/--save-dir $V\/ckpt/--save-dir $OUT/ckpt}"
case "$ARGS" in *"--save-dir $OUT/ckpt "*) ;; *) echo "the save folder was not replaced"; exit 2;; esac
EXTRA=""
if [ "$DEV" = cpu ]; then ENVS="$ENVS CUDA_VISIBLE_DEVICES="; EXTRA="--device cpu"; fi
CMD="env $ENVS OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CF_NAN_DEBUG=1 RERUN_DUMP=$D/nan_dump.pt RERUN_OUT=$OUT/records.pt RERUN_MODE=$MODE RERUN_SEED=$SEED RERUN_WEIGHTS=$WEIGHTS RERUN_STEP=13755 python3 -u $D/rerun_step.py $TRAIN $ARGS $EXTRA"
if [ "${4:-}" = check ]; then echo "$CMD"; exit 0; fi
mkdir -p "$OUT/ckpt"
export HF_TOKEN="$(cat "$WT/experiments/hf_token.txt")" HUGGING_FACE_HUB_TOKEN="$(cat "$WT/experiments/hf_token.txt")"
cd "$D" && exec $CMD
