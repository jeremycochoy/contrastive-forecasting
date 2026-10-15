#!/bin/bash
# #425 — the proof of the Moirai jobs on a GPU of elisa, before their queues.
#
# The reconstruction head of a copy of Moirai (MPM: mean/std, MPE: EWMA) reads
# the output of the transformer at each patch: the latent that the value head
# of the copy reads. The head of a run of ours reads the encoder latent. The
# head trainer and the eval read the kind off the checkpoint.
#
# With each head (the transformer head, then the linear head), parity_lin.sh
# trains the heads of MPM 100k and of MPE 100k for 1,000 steps in two ways:
#   solo  alone, with head_eval_bb.sh,
#   wave  in the queue, on one data stream with OMB 25k, a run of ours.
# It compares the loss CSVs and the final heads of each pair. Then R scores
# the wave head of each of the two checkpoints on the 4 configs of
# box_smoke.sh.
#
# Then the floor of R of each of the two checkpoints (EVAL_STRATEGY=R0) on
# the same 4 configs, beside the floor table of its scaling setup. floors.tsv
# names the setup of each run: results/per_config/floor_<setup>.csv. The same
# MASE on each config shows that the checkpoint has the scaling setup of
# that floor, so the run needs no floor of its own.
#
# The code is a deployed copy, as for a queue. Fill it one time, and after
# each code change:
#   CF425_ELISA_BASE=~/checkpoints_backup/cf-425-moirai bash deploy_elisa.sh
# Every file goes to ~/checkpoints_backup/cf-425-moirai/parity/<commit of the
# code>: test folders that no queue reads. So a new commit starts a new test,
# and a second run of one commit trains only the heads that it lacks.
#
# Usage, on elisa, in a checkout of the branch:
#   bash parity_moirai.sh [GPU] 2>&1 | tee ../results/parity_moirai.txt
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
BASE="${CF425_MOIRAI_BASE:-$HOME/checkpoints_backup/cf-425-moirai}"
CODE="${CF425_CODE:-$BASE/code}"
CK="${CF425_CK:-$HOME/checkpoints_backup/cf-412/vast_lr100x}"
COMMIT="$(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null)"
OUT="$BASE/parity/${COMMIT:-no_commit}"
SCRIPTS="$CODE/reports/2026-10-06_encoder_reconstruction/scripts"
B4="$CODE/reports/2026-08-08_rollout_depth/scripts"
JOBS="${CF425_JOBS:-$SCRIPTS/jobs.tsv}"
FLOORS="${CF425_FLOORS:-$(dirname "$HERE")/results/floors.tsv}"
FLOOR_TABLES="$(dirname "$HERE")/results/per_config"
PAIRS="${CF425_PARITY_PAIRS:-MPM:100 MPE:100}"
FILTER='^(m4_hourly/short|electricity/H/short|ett1/15T/long|us_births/D/short)$'
SHAPE="--d-model 384 --n-heads 8 --num-layers 3"
export GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 moirai proof] $*"; }

[ -f "$SCRIPTS/parity_lin.sh" ] \
  || { echo "ABORT: no code at $CODE. Run deploy_elisa.sh with CF425_ELISA_BASE=$BASE." >&2; exit 2; }
mkdir -p "$OUT/floors"
log "code $COMMIT, GPU $GPU of $(hostname), jobs: $PAIRS"

for arch in transformer linear; do
  CF425_HEAD_ARCH="$arch" CF425_PARITY_PAIRS="$PAIRS" \
    CF425_PARITY_WAVES="$PAIRS OMB:25" CF425_JOBS="$JOBS" CF425_CODE="$CODE" \
    CF425_CK="$CK" CF425_GPU="$GPU" CF425_PARITY_ROOT="$OUT/$arch/ckpt" \
    CF425_PARITY_RES="$OUT/$arch/results" \
    CF425_GPU_LOCK="$OUT/gpu_start.lock" \
    bash "$SCRIPTS/parity_lin.sh" || log "the $arch parity ended with rc=$?"
done

# "<arm> <ckpt>" of one job of the job table.
job_of(){  # <code>:<stop k>
  awk -F'\t' -v job="$1" '!/^#/ && $1 ":" $3 == job { print $2, $5 }' "$JOBS"
}

# The scaling setup of floors.tsv that names a run.
setup_of(){  # <arm>
  awk -F'\t' -v arm="$1" 'NR > 1 && index("," $3 ",", "," arm ",") { print $1 }' "$FLOORS"
}

for pair in $PAIRS; do
  read -r arm ckpt < <(job_of "$pair")
  setup="$(setup_of "$arm")"
  [ -n "$setup" ] || { log "the floor of $pair: $FLOORS names no setup for $arm"; continue; }
  tag="floor_${arm}_bb${pair#*:}k"
  WT="$CODE" EVAL_STRATEGY=R0 EVAL_DEVICE=cuda BB_GPU="$GPU" EVAL_SHARDS=1 \
    EVAL_CONFIG_FILTER="$FILTER" EVAL_EXPECT_CONFIGS=4 CF_BB_SHAPE="$SHAPE" \
    CF393_EVAL_SLOTDIR="$OUT/floors/slots" \
    bash "$B4/eval_local.sh" "$tag" "${pair#*:}" student "$CK/$ckpt" none \
    "$OUT/floors/$tag" "$OUT/floors/score_$tag.txt" >>"$OUT/floors/run.log" 2>&1 </dev/null \
    || { log "the floor of $pair FAILED: see $OUT/floors/run.log"; continue; }
  python3 "$HERE/parity_compare.py" --tables \
    "floor of ${pair%:*} ${pair#*:}k, 4 configs, against the floor $setup" \
    "$OUT/floors/$tag/gift_r0/all_results.csv" "$FLOOR_TABLES/floor_$setup.csv"
done
log "done"
