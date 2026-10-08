#!/bin/bash
# #425 — the floor of strategy R for each scaling setup of the runs: the R
# score of a head that gives the normalised value 0 and has no training
# (`EVAL_STRATEGY=R0`, `--zero-head`).
#
# Unscaled, each value of that head is the mean that normalised it. Under
# the EWMA, it is the EWMA at that value, and the EWMA reads the true horizon
# up to that value. Under the mean/std scaling, it is the loc of the context.
# So the floor is the part of an R score that the statistics give with no
# encoder, and an R score compares with the floor of its own setup.
#
# The floor reads the scaling and the context padding of the run, and no
# patch size and no weight (tests). So one checkpoint of each setup gives the
# floor of all the runs of that setup:
#   ewma_zero_pad  EWMA, zero padding (GiftEvalPretrain): BLK, OEF, MPE
#   ewma_old       EWMA, no padding (the old data): CYN, MIN, TWN, LOW, LNG,
#                  ABC
#   meanstd        mean/std, zero padding: BMS, the other O* runs, MPM
# MPM and MPE are copies of Moirai. A floor reads no latent, so each one has
# the floor of the runs of ours with its scaling (parity_moirai.sh compares
# the two on 4 configs).
#
# Writes results/floors.tsv (setup, label, arms, GM-Relative MASE) and the
# per-config table of each floor, results/per_config/floor_<setup>.csv. The
# 97 configs and the context of 1,024 values are those of the R scores.
#
# Usage, on elisa:  bash floors.sh        (FLOOR_GPU: the card, default 0)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${CF425_JOBS:-$HERE/jobs.tsv}"
MIRROR="${CF425_MIRROR:-$HOME/checkpoints_backup/cf-412/vast_lr100x}"
OUT="${CF425_FLOOR_ROOT:-$HOME/checkpoints_backup/cf-425/floors}"
RESULTS="${CF425_FLOOR_RESULTS:-$(dirname "$HERE")/results}"
EVAL_LOCAL="$HERE/../../2026-08-08_rollout_depth/scripts/eval_local.sh"
export WT="${WT:-$(cd "$HERE/../../.." && pwd)}"

# <setup> <code and stop of the checkpoint that scores it> <runs> <label>
SETUPS="
ewma_zero_pad BLK 100 BLK,OEF,MPE EWMA floor (zero padding)
ewma_old LOW 665 CYN,MIN,TWN,LOW,LNG,ABC EWMA floor (old data)
meanstd BMS 40 OMB,OCB,OCF,OMF,OAF,OWF,OWR,OWL,OBM,OBW,OAL,BMS,MPM mean/std floor
"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 floors] $*"; }

ckpt_of(){  # <code> <stop k>
  awk -F'\t' -v c="$1" -v s="$2" '!/^#/ && $1 == c && $3 == s { print $5 }' "$JOBS"
}

arms_of(){  # <codes, comma separated>
  awk -F'\t' -v codes=",$1," '!/^#/ && index(codes, "," $1 ",") { print $2 }' \
    "$JOBS" | sort -u | paste -sd,
}

mkdir -p "$OUT" "$RESULTS/per_config"
rows=()
while read -r setup code stop runs label; do
  [ -n "$setup" ] || continue
  ckpt=$(ckpt_of "$code" "$stop")
  [ -n "$ckpt" ] && [ -f "$MIRROR/$ckpt" ] \
    || { log "ABORT: no checkpoint of $code ${stop}k under $MIRROR"; exit 3; }
  dir="$OUT/floor_$setup"
  log "$setup: $code ${stop}k"
  EVAL_STRATEGY=R0 EVAL_DEVICE="${EVAL_DEVICE:-cuda}" BB_GPU="${FLOOR_GPU:-0}" \
    EVAL_SHARDS="${EVAL_SHARDS:-2}" \
    CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" \
    CF393_EVAL_SLOTDIR="${CF393_EVAL_SLOTDIR:-/tmp/cf425_floor_slots}" \
    bash "$EVAL_LOCAL" "floor_$setup" "$stop" student "$MIRROR/$ckpt" none \
    "$dir" "$OUT/score_floor_$setup.txt" \
    || { log "ABORT: the floor $setup failed. See $dir/eval_local_r0.log"; exit 4; }
  cp "$dir/gift_r0/all_results.csv" "$RESULTS/per_config/floor_$setup.csv"
  rows+=("$setup	$label	$(arms_of "$runs")	$(cat "$OUT/score_floor_$setup.txt")")
  log "$setup: $(cat "$OUT/score_floor_$setup.txt")"
done <<<"$SETUPS"
{ printf 'setup\tlabel\tarms\tgm_relative_mase\n'; printf '%s\n' "${rows[@]}"; } \
  >"$RESULTS/floors.tsv.tmp" && mv "$RESULTS/floors.tsv.tmp" "$RESULTS/floors.tsv"
log "done: $RESULTS/floors.tsv"
