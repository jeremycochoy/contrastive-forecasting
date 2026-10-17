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
# By default the scores run on the CPU, as the standard score does: about
# 3.3 hours with 4 shards, and 5 scores at the same time use 20 cores. On a
# GPU of elisa one score takes 37 minutes, but it is not the same score: for
# the standard head of BLK 200k the GM-Relative MASE is 1.126275 on the GPU
# and 1.126227 on the CPU, and the MASE of one config differs by up to 0.06%
# (b4_gpu_check.sh, 10-10). On 6 configs, the CPU of elisa and the CPU of
# the box agree to 0.00002%.
# FF_EVAL_DEVICE=cuda scores on the GPU. The waves of this work score on the
# GPU: give it to each start.
# `collect_scores.py` then writes one row for each scored (run, stop, arm)
# to `results/scores.tsv`, with its config count and its device, and the
# MASE of each config to `results/config_mase.tsv`. It compares an arm with
# a control of the same configs and the same device only.
#
# The protocol of a score is its device, its config count and its config
# filter. The score file and the eval folder of a tag have one name for each
# protocol. So the script writes the protocol to
# `heads/eval/<tag>/gift/protocol.txt` before an eval, and a start reads it
# first. A start that asks for another protocol does not take the score, and
# does not go on from the shard tables: it refuses the arm. FF_RESCORE=1
# moves the score and the eval folder to `heads/eval/<tag>/old_eval_<time>/`
# and scores again. The code of ad1984a6 wrote no protocol file: there the
# log of the runner names the device, and the eval table gives the count. A
# start with a filter refuses such a score: its filter is not known.
#
# A second start skips the work that is done. An arm with a final head does
# not train again, and an arm with a score of the same protocol is not
# scored again. A wave that stops during its training has no final head:
# the next start trains its arms again from step 1, on the same batches.
#
# One wave reads the stream at a time (`locks/stream.lock`): the Hugging Face
# token has a request limit, and the backbone run reads one stream too. A
# start that waited for the stream looks for the final heads again, so it
# trains no arm that another start trained in that time. One start scores a
# tag at a time (`locks/score_<tag>.lock`). Each wait of the script has an
# end. The wait for an eval slot is in `eval_slot.sh`: it ends after
# CF393_EVAL_SLOT_TIMEOUT seconds (one day).
#
# Every file is under FF_BASE, so a restart of elisa keeps it:
#   code/                  the code that the wave reads (deploy.sh). A
#                          deployed script reads the code folder that holds
#                          it (code_folder.sh), so a second folder
#                          (`code_v2`) can hold a fix while a wave reads the
#                          first.
#   heads/eval/<tag>/      the head of an arm, its loss CSV and its eval
#   results/               scores.tsv, config_mase.tsv, arms.tsv,
#                          score_<tag>.txt, the logs and the eval slots
#   results/waves/<wave>/  the job flags and the trainer log of one wave
#   locks/                 the stream lock, the table lock and the score
#                          locks
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
#   FF_RESCORE=1                        set a score of another protocol
#                                       aside, and score again
#   FF_CODE=<folder>                    another code folder
set -uo pipefail

RUN="${1:?usage: run_wave.sh <run> <stop k> <backbone .pth> [arm ...]}"
STOP="${2:?the stop of the backbone, in thousands of steps}"
BB="${3:?the backbone checkpoint}"
shift 3
ARMS=("$@")
[ "${#ARMS[@]}" -gt 0 ] || ARMS=(control shared_strict shared_draw heads_strict heads_draw)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
. "$HERE/code_folder.sh"
ROOT="$BASE/heads"
RES="$BASE/results"
LOCKS="$BASE/locks"
GPU="${FF_GPU:-1}"
HEAD_STEPS="${FF_HEAD_STEPS:-30000}"
TRAIN="${FF_TRAIN:-1}"
SCORE="${FF_SCORE:-1}"
DEVICE="${FF_EVAL_DEVICE:-cpu}"
FILTER="${FF_EVAL_FILTER:-}"
if [ -n "$FILTER" ]; then EXPECT="${FF_EVAL_EXPECT:-}"; else EXPECT=97; fi
# The protocol of the scores of this start.
PROTOCOL="device=$DEVICE configs=$EXPECT filter=$FILTER"
RESCORE="${FF_RESCORE:-0}"
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
# The longest waits for another start that scores the same arm, and for the
# table lock. A collect takes seconds.
SCORE_WAIT="${FF_SCORE_WAIT:-86400}"
TABLES_WAIT="${FF_TABLES_WAIT:-600}"
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
  [ -z "$FILTER" ] || JOB_ENV+=(EVAL_CONFIG_FILTER="$FILTER"
                                EVAL_EXPECT_CONFIGS="$EXPECT")
  JOB_ENV+=($(arm_env "$1"))
}

# The frequency vocabulary of the backbone: v1, v2, or none.
bb_vocab(){
  PYTHONPATH="$CODE" python3 - "$BB" <<'PY'
import sys, torch
from src.freq_embedding import vocab_of_rows
table = torch.load(sys.argv[1], map_location="cpu", weights_only=True).get(
    "freq_embedding.embedding.weight")
print("none" if table is None else vocab_of_rows(table.shape[0]))
PY
}

check_inputs(){
  local arm size family=0 vocab
  [[ "$RUN" =~ ^[a-z0-9_]+$ ]] || { log "ABORT: the run name '$RUN' must hold a-z, 0-9 and _ only"; return 1; }
  [[ "$STOP" =~ ^[0-9]+$ ]] || { log "ABORT: the stop '$STOP' is not a number of thousands of steps"; return 1; }
  for arm in "${ARMS[@]}"; do
    arm_env "$arm" >/dev/null || { log "ABORT: unknown arm '$arm'. Use control, or <shared|heads>_<strict|draw>[_m<keys>]."; return 1; }
    [ "$arm" = control ] || family=1
  done
  case "$DEVICE" in cpu|cuda) ;; *) log "ABORT: FF_EVAL_DEVICE=$DEVICE. Use cpu or cuda."; return 1 ;; esac
  [ -s "$BB" ] || { log "ABORT: no backbone at $BB"; return 1; }
  [ -f "$RUNNER" ] || { log "ABORT: no runner at $RUNNER. Run deploy.sh."; return 1; }
  [ -f "$TRAINER" ] || { log "ABORT: no shared trainer at $TRAINER. Run deploy.sh."; return 1; }
  [ -f "$COLLECT" ] || { log "ABORT: no collect script at $COLLECT. Run deploy.sh."; return 1; }
  [ -s "$CODE/experiments/hf_token.txt" ] || { log "ABORT: no Hugging Face token in $CODE/experiments"; return 1; }
  [ -d "$GIFT_EVAL" ] || { log "ABORT: no GIFT-Eval data at $GIFT_EVAL"; return 1; }
  [[ "$EXPECT" =~ ^[0-9]+$ ]] \
    || { log "ABORT: FF_EVAL_FILTER needs FF_EVAL_EXPECT, the count of its configs"; return 1; }
  size=$(stat -c %s "$SN_REF" 2>/dev/null || echo missing)
  [ "$size" = "$SN_BYTES" ] || { log "ABORT: the seasonal-naive reference $SN_REF is $size bytes, want $SN_BYTES"; return 1; }
  # A family arm needs the labels of the vocabulary v2. With v1, the stream
  # gives no label to a row of 4 seconds, of 6 hours, of a month, a quarter
  # or a year. Such a row would train another member, with no error.
  if [ "$family" = 1 ]; then
    vocab="$(bb_vocab 2>/dev/null)"
    [ "$vocab" = v2 ] || { log "ABORT: a family arm needs a backbone with the frequency vocabulary v2. $BB has '${vocab:-no table that loads}'."; return 1; }
  fi
}

# Wait for the table lock on fd 8. A stopped process can hold it, so the
# wait has an end.
table_lock(){
  flock -w "$TABLES_WAIT" 8 \
    || { log "the table lock $LOCKS/tables.lock is held after ${TABLES_WAIT} s"; return 1; }
}

# One line of results/arms.tsv for each arm of the wave: collect_scores.py
# reads the run, the stop and the arm of a tag there.
note_arms(){
  local arm tag table="$RES/arms.tsv"
  (
    table_lock || exit 1
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

# The arms with no final head, in TODO.
todo_arms(){
  local arm
  TODO=()
  for arm in "${ARMS[@]}"; do
    has_head "$(tag_of "$arm")" || TODO+=("$arm")
  done
}

# Train the arms with no final head, in one process. Returns 1 when an arm
# has no final head after it.
train_wave(){
  local lk rc
  todo_arms
  if [ "${#TODO[@]}" -eq 0 ]; then
    log "train SKIP: each arm has its final head"; return 0
  fi
  exec {lk}>>"$STREAM_LOCK"
  flock -n "$lk" || log "waiting: another wave reads the data stream ($STREAM_LOCK)"
  if flock -w "$STREAM_WAIT" "$lk"; then
    train_arms; rc=$?
  else
    log "ABORT: another wave reads the data stream after ${STREAM_WAIT} s"; rc=1
  fi
  # Each way out frees the stream: the scores read no stream.
  exec {lk}>&-
  return "$rc"
}

# The training, with the stream lock held.
train_arms(){
  local arm tag dir free start rc
  # Another start can train arms while this one waits for the stream. A head
  # that trains again would not be the head of the score on disk.
  todo_arms
  if [ "${#TODO[@]}" -eq 0 ]; then
    log "train SKIP: another start trained each arm"; return 0
  fi
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
  mkdir -p "$RES/waves"
  dir=$(mktemp -d "$RES/waves/${RUN}_bb${STOP}k_$(date '+%m%d_%H%M%S')_XXXX") || return 1
  for arm in "${TODO[@]}"; do
    tag="$(tag_of "$arm")"
    job_env "$arm"
    env "${JOB_ENV[@]}" CF_HEAD_ARGV_TO="$dir/jobs.jsonl" \
      bash "$RUNNER" "$tag" "$BB" student "$HEAD_STEPS" \
      >>"$RES/heads.log" 2>&1 </dev/null \
      || { log "ABORT: no head flags for $tag. See $RES/heads.log."; return 1; }
  done
  [ "$(grep -c . "$dir/jobs.jsonl" 2>/dev/null)" = "${#TODO[@]}" ] \
    || { log "ABORT: $dir/jobs.jsonl does not hold ${#TODO[@]} jobs"; return 1; }
  log "train: ${#TODO[@]} arms, $HEAD_STEPS steps, GPU $GPU, code $(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null): ${TODO[*]} -> $dir/train.log"
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
  log "train rc=$rc"
  for arm in "${TODO[@]}"; do
    has_head "$(tag_of "$arm")" || { log "no final head for $arm after the trainer"; rc=1; }
  done
  return "$rc"
}

# The protocol of the eval work of a tag, or nothing for a tag with none: no
# table of a shard, no table of the eval and no score. The code of ad1984a6
# wrote no protocol file. There the log of the runner names the device of
# each eval start. The table of a complete eval gives the config count: 0
# for a score with no table.
eval_protocol(){  # <tag>
  local dir="$ROOT/eval/$1" devices text
  compgen -G "$dir/gift/shard_*/all_results.csv" >/dev/null \
    || [ -e "$dir/gift/all_results.csv" ] || has_score "$1" || return 0
  if [ -e "$dir/gift/protocol.txt" ]; then
    text="$(cat "$dir/gift/protocol.txt")"
    echo "${text:-a protocol file with no text}"
    return 0
  fi
  devices=$(sed -n 's/.*\] eval start (.*, \([a-z]*\))$/\1/p' "$dir/stop.log" 2>/dev/null \
    | sort -u | paste -sd+ -)
  printf 'device=%s' "${devices:-unknown}"
  if has_score "$1" || [ -e "$dir/gift/all_results.csv" ]; then
    printf ' configs=%s' "$(tail -n +2 "$dir/gift/all_results.csv" 2>/dev/null | wc -l)"
  fi
  echo
}

# The eval work of a protocol is the work of this start. The work of the
# code of ad1984a6 names no filter, and no count before its end. In a start
# with no filter, the fields that it names decide. A start with a filter
# cannot know that the work had the same filter.
same_protocol(){  # <protocol>
  [ "$1" = "$PROTOCOL" ] && return 0
  [[ -z "$FILTER" && "$1" != *" filter="* && "$PROTOCOL " == "$1 "* ]]
}

# Move the score and the eval folder of a tag to a folder of their own.
# Nothing is deleted. The score goes first: a stop between the two leaves no
# score with no eval table.
set_aside(){  # <tag>
  local dir="$ROOT/eval/$1" old
  old="$dir/old_eval_$(date '+%m%d_%H%M%S')"
  mkdir "$old" || return 1
  [ ! -e "$RES/score_$1.txt" ] || mv "$RES/score_$1.txt" "$old/" || return 1
  [ ! -d "$dir/gift" ] || mv "$dir/gift" "$old/gift" || return 1
  log "set aside: the score and the eval files of $1 -> $old"
}

# The B4 score of one arm. Returns 0 when the arm has its score of this
# protocol, NO_SCORE when it gets none and the log says why, else the code
# of the runner.
NO_SCORE=90
score_arm(){  # <arm>
  local arm="$1" tag lk have
  tag="$(tag_of "$arm")"
  # One start scores a tag at a time: two evals of one tag would write the
  # same shard tables. The eval keeps the lock too, also after a stop of
  # this script.
  exec {lk}>>"$LOCKS/score_$tag.lock"
  flock -n "$lk" || log "waiting: another start scores $arm ($LOCKS/score_$tag.lock)"
  flock -w "$SCORE_WAIT" "$lk" \
    || { log "no score for $arm: another start scores it after ${SCORE_WAIT} s"; return "$NO_SCORE"; }
  have="$(eval_protocol "$tag")"
  if [ -n "$have" ] && ! same_protocol "$have"; then
    if [ "$RESCORE" != 1 ]; then
      log "score $arm REFUSED: its eval files are of '$have', and this start asks for '$PROTOCOL'. Give the same device and filter. Or set FF_RESCORE=1: it sets these files aside and scores again."
      return "$NO_SCORE"
    fi
    set_aside "$tag" || return 1
  fi
  has_score "$tag" && return 0
  has_head "$tag" || { log "no score for $arm: it has no final head"; return "$NO_SCORE"; }
  mkdir -p "$ROOT/eval/$tag/gift" \
    && printf '%s\n' "$PROTOCOL" >"$ROOT/eval/$tag/gift/protocol.txt" \
    || return 1
  job_env "$arm"
  env "${JOB_ENV[@]}" bash "$RUNNER" "$tag" "$BB" student "$HEAD_STEPS" \
    >>"$RES/scores.log" 2>&1 </dev/null
}

# The B4 score of each arm, at the same time. Returns 1 when an arm ends
# with no score of this protocol.
score_wave(){
  local arm pids=() i fail=0
  log "score: ${ARMS[*]} ($PROTOCOL, $SHARDS shards, $SLOTS at a time)"
  for arm in "${ARMS[@]}"; do
    score_arm "$arm" &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    arm="${ARMS[$i]}"
    wait "${pids[$i]}"
    case $? in
      0) log "score $arm: $(cat "$RES/score_$(tag_of "$arm").txt" 2>/dev/null)" ;;
      "$NO_SCORE") fail=1 ;;
      *) log "score $arm FAILED. See $RES/scores.log."; fail=1 ;;
    esac
  done
  return "$fail"
}

collect(){
  ( table_lock || exit 1
    PYTHONPATH="$CODE" python3 "$COLLECT" --base "$BASE"
  ) 8>>"$LOCKS/tables.lock"
}

mkdir -p "$ROOT" "$RES" "$LOCKS" || exit 2
check_inputs || exit 2
note_arms || { log "ABORT: no line of $RES/arms.tsv for the arms"; exit 2; }
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
