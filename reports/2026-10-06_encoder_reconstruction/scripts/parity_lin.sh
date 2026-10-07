#!/bin/bash
# #425 — the parity test of a linear head on the box, before the linear queue.
#
# Two jobs each train a linear head of 1,000 steps in two ways:
#   solo  alone, with head_eval_bb.sh,
#   wave  in the linear queue, on one data stream with other jobs.
# OMB 25k (patch sizes 8 to 128, mean/std) trains beside BLK 200k, BMS 40k
# and OEF 40k on GiftEvalPretrain. LOW 665k (one size, EWMA) trains beside
# TWN 100k on the old data. So the two waves hold one job of each kind of
# run. parity_compare.py compares the loss CSVs and the final heads of each
# pair. Then R scores the wave heads of the two jobs on the 4 configs of
# box_smoke.sh.
#
# Every file goes to test folders that the linear queue never reads. The
# waves take the GPU lock of the queues, so the test can run beside a queue.
# The scores take their own eval slot: they are short.
#
# Usage, on the box:  bash parity_lin.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${CF425_CODE:-/workspace/cf-425-lin}"
CK="${CF425_CK:-/workspace/ckpt}"
OUT="${CF425_PARITY_ROOT:-$CK/cf-425-lin/parity}"
RES="${CF425_PARITY_RES:-/workspace/results/cf-425-lin/parity}"
GPU="${CF425_GPU:-0}"
STEPS="${CF425_PARITY_STEPS:-1000}"
B4="$CODE/reports/2026-08-08_rollout_depth/scripts"
FILTER='^(m4_hourly/short|electricity/H/short|ett1/15T/long|us_births/D/short)$'
SHAPE="--d-model 384 --n-heads 8 --num-layers 3"
export GIFT_EVAL="${GIFT_EVAL:-/workspace/gift-eval-data}"
SEED=20260722
# The jobs with a solo run, and all the jobs of the waves: "<code>:<stop k>".
PAIRS="OMB:25 LOW:665"
WAVES="$PAIRS BLK:200 BMS:40 OEF:40 TWN:100"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 linear parity] $*"; }

# "<arm> <ckpt>" of one job of jobs.tsv.
job_of(){  # <code>:<stop k>
  awk -F'\t' -v job="$1" '!/^#/ && $1 ":" $3 == job { print $2, $5 }' "$HERE/jobs.tsv"
}

tag_of(){  # <code>:<stop k>
  local arm ckpt
  read -r arm ckpt < <(job_of "$1")
  echo "${arm}_bb${1#*:}k_h30k_recon_lin"
}

head_of(){ echo "$OUT/$1/eval/$2/qhead_${2}_s${SEED}_$3"; }   # <solo|queue> <tag> <file end>

mkdir -p "$OUT" "$RES"
awk -F'\t' -v want=" $WAVES " '/^#code/ || index(want, " " $1 ":" $3 " ")' \
  "$HERE/jobs.tsv" >"$RES/jobs.tsv"

pids=()
for pair in $PAIRS; do
  read -r arm ckpt < <(job_of "$pair")
  tag=$(tag_of "$pair")
  if [ -f "$(head_of solo "$tag" final.pth)" ]; then
    log "solo $pair: it ended, so it does not start again"; continue
  fi
  log "solo $pair: $STEPS steps, code $(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null)"
  WT="$CODE" CF373_ROOT="$OUT/solo" CF_RESULTS="$RES/solo" CF_STOP_K="${pair#*:}" \
    CF_RECONSTRUCTION=encoder CF_HEAD_ARCH=linear HEAD_SAVE_EVERY=1000000 \
    CF_SKIP_EVAL=1 CF_BB_SHAPE="$SHAPE" BB_GPU="$GPU" HEAD_VRAM_MIB=4000 \
    GPU_GATE_LOCKDIR="/tmp/cf425_parity_lin_${pair%:*}" \
    bash "$B4/head_eval_bb.sh" "$tag" "$CK/$ckpt" student "$STEPS" \
    >"$RES/solo_${pair%:*}.out" 2>&1 </dev/null &
  pids+=($!)
done

log "waves: $(grep -vc '^#' "$RES/jobs.tsv") jobs, $STEPS steps"
CF425_HEAD_ARCH=linear CF425_CODE="$CODE" CF425_CK="$CK" CF425_JOBS="$RES/jobs.tsv" \
  CF425_HEAD_STEPS="$STEPS" CF425_SCORE=0 CF425_ROOT="$OUT/queue" \
  CF425_RES="$RES/queue" CF425_LANE_STAGGER=30 CF425_GPU="$GPU" \
  bash "$HERE/queue.sh" >"$RES/queue.out" 2>&1
log "waves rc=$?"
for pid in "${pids[@]}"; do wait "$pid" || log "a solo run failed: see $RES/solo_*.out"; done

csvs=(); heads=()
for pair in $PAIRS; do
  tag=$(tag_of "$pair")
  label="${pair%:*} ${pair#*:}k"
  csvs+=("$label" "$(head_of queue "$tag" losses.csv)" "$(head_of solo "$tag" losses.csv)")
  heads+=("$label" "$(head_of queue "$tag" final.pth)" "$(head_of solo "$tag" final.pth)")
  log "solo $label: $(grep -h 'sps' "$OUT/solo/eval/$tag/stop.log" | tail -1)"
done
{ python3 "$HERE/parity_compare.py" "${csvs[@]}"
  python3 "$HERE/parity_compare.py" --heads "${heads[@]}"; } | tee "$RES/compare.txt"
grep -h '^\[shared\]' "$RES"/queue/waves/*/train.log | grep -v ' 1 steps' | tail -4

for pair in $PAIRS; do
  read -r arm ckpt < <(job_of "$pair")
  tag=$(tag_of "$pair")
  WT="$CODE" EVAL_STRATEGY=R EVAL_DEVICE=cuda BB_GPU="$GPU" EVAL_SHARDS=1 \
    EVAL_CONFIG_FILTER="$FILTER" EVAL_EXPECT_CONFIGS=4 CF_BB_SHAPE="$SHAPE" \
    CF393_EVAL_SLOTDIR=/tmp/cf425_parity_lin_slots \
    bash "$B4/eval_local.sh" "$tag" "${pair#*:}" student "$CK/$ckpt" \
    "$(head_of queue "$tag" final.pth)" "$OUT/r4/$tag" "$RES/score_r4_$tag.txt" \
    >>"$RES/r4.out" 2>&1 </dev/null
  log "R, 4 configs, linear head of ${pair%:*} ${pair#*:}k: $(cat "$RES/score_r4_$tag.txt" 2>/dev/null || echo FAILED)"
done
