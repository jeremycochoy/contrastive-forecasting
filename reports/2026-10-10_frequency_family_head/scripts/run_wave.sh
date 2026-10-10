#!/bin/bash
# freq_family — one wave: the forecast heads of the arms of one backbone
# checkpoint, trained together on one data stream, then the B4 score of each
# head.
#
# The arms:
#   control                the standard B4 head, trained again in this wave
#   <body>_<rule>          a frequency family (`src/freq_family.py`): the
#                          body `shared` or `heads`, the rule `strict` or
#                          `draw`
#   <body>_<rule>_m<keys>  the same with another member list, the keys
#                          joined with `-`. `heads_strict_m16` has one
#                          member, so it trains as the control.
# With no arm, the wave holds the control and the 4 family arms.
#
# Each arm keeps the flags of the standard B4 head of a GiftEvalPretrain
# backbone (`head_eval_bb.sh`: 30,000 steps, batch 256, lr 1e-3, head seed
# 20260722 and so on). The family flags are the only change. One process
# (`train_forecasting_heads_shared.py`) trains the arms of the wave: each
# batch of the stream goes to each arm in turn. So the arms of one wave read
# the same batches, and a family is compared with the control of its wave.
#
# Then each head gets its B4 score: the 97 GIFT-Eval configs with a context
# of 1,024 values, the protocol of the standard score. The scores of a wave
# run at the same time, and the eval slots of the base folder limit them.
#
# The scores run on the CPU, as the standard score does: about 3.3 hours
# with 4 shards, and 5 scores at the same time use 20 cores. On a GPU of
# elisa one score takes 37 minutes, but it is not the same score: for the
# standard head of BLK 200k the GM-Relative MASE is 1.126275 on the GPU and
# 1.126227 on the CPU, and the MASE of one config differs by up to 0.06%
# (b4_gpu_check.sh, 10-10). On 6 configs, the CPU of elisa and the CPU of
# the box agree to 0.00002%.
# FF_EVAL_DEVICE=cuda scores on the GPU. Score each arm of one wave on one
# device.
# `collect_scores.py` then writes one row for each scored (run, stop, arm)
# to `results/scores.tsv`, and the MASE of each config to
# `results/config_mase.tsv`.
#
# A second start skips the work that is done. An arm with a final head does
# not train again, and an arm with a score file is not scored again. A wave
# that stops during its training has no final head: the next start trains
# its arms again from step 1, on the same batches.
#
# One wave reads the stream at a time (`locks/stream.lock`): the Hugging Face
# token has a request limit, and the backbone run reads one stream too. Each
# wait of the script has an end.
#
# Every file is under FF_BASE, so a restart of elisa keeps it:
#   code/                  the code that the wave reads (deploy.sh)
#   heads/eval/<tag>/      the head of an arm, its loss CSV and its eval
#   results/               scores.tsv, config_mase.tsv, arms.tsv,
#                          score_<tag>.txt, the logs and the eval slots
#   results/waves/<wave>/  the job flags and the trainer log of one wave
#   locks/                 the stream lock and the table lock
# A tag is <run>_bb<stop>k_h<head steps>_<arm>, with `ff_` before a family
# arm: blk_bb200k_h30k_control, blk_bb200k_h30k_ff_shared_strict.
#
# Usage, on elisa, from the code folder:
#   bash run_wave.sh <run> <stop k> <backbone .pth> [arm ...]
#   FF_SCORE=0 bash run_wave.sh ...     train the heads, and score nothing
#   FF_TRAIN=0 bash run_wave.sh ...     train nothing: score the heads that
#                                       exist
#   FF_GPU=<N>                          the GPU of the wave (default 1)
#   FF_EVAL_DEVICE=cpu|cuda             the device of the scores (default cpu)
#   FF_EVAL_SHARDS, FF_EVAL_SLOTS       the shards of one score, and the
#                                       scores that run at the same time
#   FF_HEAD_STEPS=500                   a short test wave (tags `_h500_`)
#   FF_EVAL_FILTER, FF_EVAL_EXPECT      a test score on a few configs: the
#                                       regex of the configs, and their count
set -uo pipefail

RUN="${1:?usage: run_wave.sh <run> <stop k> <backbone .pth> [arm ...]}"
STOP="${2:?the stop of the backbone, in thousands of steps}"
BB="${3:?the backbone checkpoint}"
shift 3
ARMS=("$@")
[ "${#ARMS[@]}" -gt 0 ] || ARMS=(control shared_strict shared_draw heads_strict heads_draw)

BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
CODE="${FF_CODE:-$BASE/code}"
ROOT="$BASE/heads"
RES="$BASE/results"
LOCKS="$BASE/locks"
GPU="${FF_GPU:-1}"
HEAD_STEPS="${FF_HEAD_STEPS:-30000}"
TRAIN="${FF_TRAIN:-1}"
SCORE="${FF_SCORE:-1}"
DEVICE="${FF_EVAL_DEVICE:-cpu}"
SHARDS="${FF_EVAL_SHARDS:-4}"
# On the CPU, 5 scores of 4 shards use 20 of the 32 cores of elisa. On the
# GPU, 2 scores run at the same time.
if [ "$DEVICE" = cuda ]; then SLOTS="${FF_EVAL_SLOTS:-2}"; else SLOTS="${FF_EVAL_SLOTS:-5}"; fi
# One wave reads the stream at a time. A wave of another base folder (the
# test wave) names the lock of the waves here, so it waits for them.
STREAM_LOCK="${FF_STREAM_LOCK:-$LOCKS/stream.lock}"
STREAM_WAIT="${FF_STREAM_WAIT:-86400}"
VRAM="${FF_WAVE_VRAM_MIB:-8000}"
VRAM_WAIT="${FF_VRAM_WAIT:-7200}"
RUNNER="${FF_RUNNER:-$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh}"
TRAINER="${FF_TRAINER:-$CODE/experiments/2026-04-13_gift-eval/scripts/train_forecasting_heads_shared.py}"
COLLECT="${FF_COLLECT:-$CODE/reports/2026-10-10_frequency_family_head/scripts/collect_scores.py}"
GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}"
# The seasonal-naive reference of the GM-Relative MASE: the first path that
# eval_gift_eval_official.py reads. Git does not hold it.
SN_REF="${FF_SN_REF:-$HOME/workspaces/gift-eval/results/seasonal_naive/all_results.csv}"
SN_BYTES=24831
# The backbone shape of BLK and of abc_gift.
SHAPE="--d-model 384 --n-heads 8 --num-layers 3"
# The knobs of head_eval_bb.sh that this script sets for each arm. A value
# of the caller must not reach an arm: a control with CF_FREQ_FAMILY would
# be a family.
unset CF_FREQ_FAMILY CF_FREQ_FAMILY_RULE CF_FREQ_FAMILY_MEMBERS \
  CF_RECONSTRUCTION CF_HEAD_ARCH CF_HEAD_ARGV_TO CF_SKIP_EVAL EVAL_STRATEGY \
  EVAL_CONFIG_FILTER EVAL_EXPECT_CONFIGS HEAD_SEED

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [freq_family $RUN ${STOP}k] $*"; }

# 30000 -> 30k, 500 -> 500: the head steps in a tag.
if [ $(( HEAD_STEPS % 1000 )) -eq 0 ]; then HK="$(( HEAD_STEPS / 1000 ))k"; else HK="$HEAD_STEPS"; fi

# The family knobs of head_eval_bb.sh for an arm, or 1 for an unknown arm.
arm_env(){  # <arm>
  [ "$1" = control ] && return 0
  [[ "$1" =~ ^(shared|heads)_(strict|draw)(_m([0-9]+(-[0-9]+)*))?$ ]] || return 1
  echo "CF_FREQ_FAMILY=${BASH_REMATCH[1]} CF_FREQ_FAMILY_RULE=${BASH_REMATCH[2]}"
  [ -z "${BASH_REMATCH[4]}" ] || echo "CF_FREQ_FAMILY_MEMBERS=${BASH_REMATCH[4]//-/,}"
}

tag_of(){  # <arm>
  if [ "$1" = control ]; then echo "${RUN}_bb${STOP}k_h${HK}_control"
  else echo "${RUN}_bb${STOP}k_h${HK}_ff_$1"; fi
}

has_head(){ compgen -G "$ROOT/eval/$1/*_final.pth" >/dev/null; }   # <tag>
has_score(){ [ -s "$RES/score_$1.txt" ]; }                          # <tag>

# The environment of one head_eval_bb.sh call of an arm, in JOB_ENV.
job_env(){  # <arm>
  JOB_ENV=(WT="$CODE" CF373_ROOT="$ROOT" CF_RESULTS="$RES" CF_STOP_K="$STOP"
           GIFT_EVAL="$GIFT_EVAL" EVAL_SHARDS="$SHARDS" EVAL_DEVICE="$DEVICE"
           CF393_EVAL_SLOTS="$SLOTS" CF393_EVAL_SLOTDIR="$RES/evalslots"
           BB_GPU="$GPU" GPU_GATE_LOCKDIR="$LOCKS" CF_BB_SHAPE="$SHAPE")
  [ -z "${FF_HEAD_SAVE_EVERY:-}" ] || JOB_ENV+=(HEAD_SAVE_EVERY="$FF_HEAD_SAVE_EVERY")
  [ -z "${FF_HEAD_LOG_EVERY:-}" ] || JOB_ENV+=(HEAD_LOG_EVERY="$FF_HEAD_LOG_EVERY")
  [ -z "${FF_EVAL_FILTER:-}" ] || JOB_ENV+=(EVAL_CONFIG_FILTER="$FF_EVAL_FILTER"
                                            EVAL_EXPECT_CONFIGS="${FF_EVAL_EXPECT:-}")
  JOB_ENV+=($(arm_env "$1"))
}

check_inputs(){
  local arm size
  [[ "$RUN" =~ ^[a-z0-9_]+$ ]] || { log "ABORT: the run name '$RUN' must hold a-z, 0-9 and _ only"; return 1; }
  [[ "$STOP" =~ ^[0-9]+$ ]] || { log "ABORT: the stop '$STOP' is not a number of thousands of steps"; return 1; }
  for arm in "${ARMS[@]}"; do
    arm_env "$arm" >/dev/null || { log "ABORT: unknown arm '$arm'. Use control, or <shared|heads>_<strict|draw>[_m<keys>]."; return 1; }
  done
  [ -s "$BB" ] || { log "ABORT: no backbone at $BB"; return 1; }
  [ -f "$RUNNER" ] || { log "ABORT: no runner at $RUNNER. Run deploy.sh."; return 1; }
  [ -f "$TRAINER" ] || { log "ABORT: no shared trainer at $TRAINER. Run deploy.sh."; return 1; }
  [ -f "$COLLECT" ] || { log "ABORT: no collect script at $COLLECT. Run deploy.sh."; return 1; }
  [ -s "$CODE/experiments/hf_token.txt" ] || { log "ABORT: no Hugging Face token in $CODE/experiments"; return 1; }
  [ -d "$GIFT_EVAL" ] || { log "ABORT: no GIFT-Eval data at $GIFT_EVAL"; return 1; }
  [ -z "${FF_EVAL_FILTER:-}" ] || [[ "${FF_EVAL_EXPECT:-}" =~ ^[0-9]+$ ]] \
    || { log "ABORT: FF_EVAL_FILTER needs FF_EVAL_EXPECT, the count of its configs"; return 1; }
  size=$(stat -c %s "$SN_REF" 2>/dev/null || echo missing)
  [ "$size" = "$SN_BYTES" ] || { log "ABORT: the seasonal-naive reference $SN_REF is $size bytes, want $SN_BYTES"; return 1; }
}

# One line of results/arms.tsv for each arm of the wave: collect_scores.py
# reads the run, the stop and the arm of a tag there.
note_arms(){
  local arm tag table="$RES/arms.tsv"
  (
    flock 8
    [ -s "$table" ] || printf 'run\tstop_k\tarm\ttag\thead_steps\tbackbone\n' >"$table"
    for arm in "${ARMS[@]}"; do
      tag="$(tag_of "$arm")"
      awk -F'\t' -v t="$tag" '$4 == t { found = 1 } END { exit !found }' "$table" \
        || printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN" "$STOP" "$arm" "$tag" "$HEAD_STEPS" "$BB" >>"$table"
    done
  ) 8>>"$LOCKS/tables.lock"
}

gpu_free(){  # MiB, or nothing with no nvidia-smi
  nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits \
    2>/dev/null | head -1 | tr -dc 0-9
}

# Train the arms with no final head, in one process. Returns 1 when an arm
# has no final head after it.
train_wave(){
  local todo=() arm tag dir lk free start rc
  for arm in "${ARMS[@]}"; do
    has_head "$(tag_of "$arm")" || todo+=("$arm")
  done
  if [ "${#todo[@]}" -eq 0 ]; then
    log "train SKIP: each arm has its final head"; return 0
  fi
  mkdir -p "$RES/waves"
  dir=$(mktemp -d "$RES/waves/${RUN}_bb${STOP}k_$(date '+%m%d_%H%M%S')_XXXX") || return 1
  for arm in "${todo[@]}"; do
    tag="$(tag_of "$arm")"
    job_env "$arm"
    env "${JOB_ENV[@]}" CF_HEAD_ARGV_TO="$dir/jobs.jsonl" \
      bash "$RUNNER" "$tag" "$BB" student "$HEAD_STEPS" \
      >>"$RES/heads.log" 2>&1 </dev/null \
      || { log "ABORT: no head flags for $tag. See $RES/heads.log."; return 1; }
  done
  [ "$(grep -c . "$dir/jobs.jsonl" 2>/dev/null)" = "${#todo[@]}" ] \
    || { log "ABORT: $dir/jobs.jsonl does not hold ${#todo[@]} jobs"; return 1; }

  exec {lk}>>"$STREAM_LOCK"
  flock -n "$lk" || log "waiting: another wave reads the data stream ($STREAM_LOCK)"
  flock -w "$STREAM_WAIT" "$lk" \
    || { log "ABORT: another wave reads the data stream after ${STREAM_WAIT} s"; return 1; }
  start=$SECONDS
  while :; do
    free=$(gpu_free)
    { [ -z "$free" ] || [ "$free" -ge "$VRAM" ]; } && break
    if (( SECONDS - start >= VRAM_WAIT )); then
      log "ABORT: GPU $GPU has $free MiB free after ${VRAM_WAIT} s. A wave needs $VRAM."
      return 1
    fi
    (( (SECONDS - start) % 600 < 30 )) && log "waiting: GPU $GPU has $free MiB free, a wave needs $VRAM"
    sleep 30
  done
  log "train: ${#todo[@]} arms, $HEAD_STEPS steps, GPU $GPU, code $(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null): ${todo[*]} -> $dir/train.log"
  # The trainer keeps the stream lock too: it reads the stream, also after
  # a stop of this script.
  env PYTHONPATH="$CODE" CUDA_VISIBLE_DEVICES="$GPU" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS="${FF_TRAIN_THREADS:-4}" MKL_NUM_THREADS="${FF_TRAIN_THREADS:-4}" \
    HF_TOKEN="$(cat "$CODE/experiments/hf_token.txt")" \
    HUGGING_FACE_HUB_TOKEN="$(cat "$CODE/experiments/hf_token.txt")" \
    python3 -u "$TRAINER" --jobs "$dir/jobs.jsonl" >>"$dir/train.log" 2>&1 \
    </dev/null
  rc=$?
  exec {lk}>&-
  log "train rc=$rc"
  for arm in "${todo[@]}"; do
    has_head "$(tag_of "$arm")" || { log "no final head for $arm after the trainer"; rc=1; }
  done
  return "$rc"
}

# The B4 score of each arm with a head and no score, at the same time.
# Returns 1 when a score fails.
score_wave(){
  local arm tag pids=() names=() i fail=0
  for arm in "${ARMS[@]}"; do
    tag="$(tag_of "$arm")"
    has_score "$tag" && continue
    has_head "$tag" || { log "no score for $arm: it has no final head"; fail=1; continue; }
    job_env "$arm"
    env "${JOB_ENV[@]}" bash "$RUNNER" "$tag" "$BB" student "$HEAD_STEPS" \
      >>"$RES/scores.log" 2>&1 </dev/null &
    pids+=($!); names+=("$arm")
  done
  [ "${#pids[@]}" -eq 0 ] || log "score: ${names[*]} ($DEVICE, $SHARDS shards, $SLOTS at a time)"
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "score ${names[$i]}: $(cat "$RES/score_$(tag_of "${names[$i]}").txt" 2>/dev/null)"
    else
      log "score ${names[$i]} FAILED. See $RES/scores.log."; fail=1
    fi
  done
  return "$fail"
}

collect(){
  ( flock 8
    PYTHONPATH="$CODE" python3 "$COLLECT" --base "$BASE"
  ) 8>>"$LOCKS/tables.lock"
}

mkdir -p "$ROOT" "$RES" "$LOCKS" || exit 2
check_inputs || exit 2
note_arms
rc=0
if [ "$TRAIN" = 1 ]; then
  train_wave || rc=1
fi
if [ "$SCORE" = 1 ]; then
  score_wave || rc=1
  collect || { log "collect FAILED"; rc=1; }
fi
log "WAVE_END rc=$rc"
exit "$rc"
