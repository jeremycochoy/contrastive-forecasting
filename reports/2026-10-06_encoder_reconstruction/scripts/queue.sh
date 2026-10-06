#!/bin/bash
# #425 — the reconstruction queue on the box: one reconstruction head and one
# reconstruction score for each row of jobs.tsv.
#
# CF425_LANES lanes train heads on the one GPU at the same time. A lane trains
# a head (`head_eval_bb.sh`, CF_SKIP_EVAL=1), starts its score in the
# background, and takes the next job. A score waits for one of
# CF393_EVAL_SLOTS slots, so the heads never wait for a score. The scores run
# on the GPU (CF425_EVAL_DEVICE): strategy R reads a batch of windows in one
# pass, and one CPU core reads about 18 windows a second.
#
# Every head and every score keeps the B4 settings of `head_eval_bb.sh`: the
# head architecture, 30,000 steps, batch 256, lr 1e-3, head seed 20260722,
# the scaling of the run, one head per patch size, and the 97 configs with a
# context of 1,024 values. Two things differ, and neither changes a weight:
# the head writes no snapshot every 5,000 steps (the disk holds 16 GB), and
# the score reads the true horizon (strategy R).
#
# A job is done when its score file exists. A lane takes a job only with a
# `flock` on the job's lock file, and the lock passes to the background score.
# So no two processes run one job, and a restart skips the jobs that an
# earlier queue still runs. A job that fails CF425_TRIES times stays failed.
#
# Order: jobs.tsv by tier, then by row. Tier 1 holds the first and the last
# stop of each run, so each run has its two ends before the stops between.
#
# Usage, on the box:
#   nohup setsid bash queue.sh >>/workspace/results/cf-425/queue.log 2>&1 &
#   CF425_DRY_RUN=1 bash queue.sh     # the job order and the input check only
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${CF425_JOBS:-$HERE/jobs.tsv}"
CODE="${CF425_CODE:-/workspace/cf-425}"
CK="${CF425_CK:-/workspace/ckpt}"
ROOT="${CF425_ROOT:-$CK/cf-425/recon}"
RES="${CF425_RES:-/workspace/results/cf-425}"
LANES="${CF425_LANES:-3}"
TRIES="${CF425_TRIES:-2}"
STAGGER="${CF425_LANE_STAGGER:-90}"
HEAD_STEPS="${CF425_HEAD_STEPS:-30000}"
RUNNER="${CF425_RUNNER:-$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh}"
LOCKS="$RES/locks"
FAILED="$RES/failed"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#425 queue] $*"; }

tag_of(){ echo "${1}_bb${2}k_h30k_recon"; }   # <arm> <stop k>

# The jobs in queue order: "<tier> <row> <code> <arm> <stop k> <ckpt> <bytes>".
ordered_jobs(){
  awk -F'\t' '!/^#/ && NF >= 6 { print $4, NR, $1, $2, $3, $5, $6 }' "$JOBS" \
    | sort -k1,1n -k2,2n
}

# Each input on the box with the byte size of its elisa copy, or ABORT.
check_inputs(){
  local bad=0 tier row code arm stop ckpt bytes size
  while read -r tier row code arm stop ckpt bytes; do
    size=$(stat -c %s "$CK/$ckpt" 2>/dev/null || echo missing)
    [ "$size" = "$bytes" ] && continue
    log "INPUT $code ${stop}k: $CK/$ckpt is $size bytes, want $bytes"
    bad=$(( bad + 1 ))
  done < <(ordered_jobs)
  [ "$bad" -eq 0 ] || { log "ABORT: $bad input(s) not on the box. Run stage_inputs.sh on elisa."; return 1; }
}

fail_count(){ ls "$FAILED/$1".* 2>/dev/null | wc -l; }

mark_failed(){  # <tag> <what>
  mkdir -p "$FAILED"
  echo "$2 $(date '+%m-%d %H:%M:%S')" >"$FAILED/$1.$(( $(fail_count "$1") + 1 ))"
}

# The environment of one head_eval_bb.sh call.
job_env(){  # <lane> <stop k>
  echo WT="$CODE" CF373_ROOT="$ROOT" CF_RESULTS="$RES" CF_STOP_K="$2" \
    GIFT_EVAL="${GIFT_EVAL:-/workspace/gift-eval-data}" \
    EVAL_SHARDS="${CF425_EVAL_SHARDS:-2}" \
    EVAL_DEVICE="${CF425_EVAL_DEVICE:-cuda}" \
    CF393_EVAL_SLOTS="${CF425_EVAL_SLOTS:-2}" \
    CF393_EVAL_SLOTDIR=/tmp/cf425_evalslots \
    BB_GPU=0 HEAD_VRAM_MIB="${CF425_HEAD_VRAM_MIB:-9000}" \
    GPU_GATE_LOCKDIR="/tmp/cf425_lane$1" \
    CF_RECONSTRUCTION=encoder HEAD_SAVE_EVERY=1000000
}

# The score of one job, in the background. It holds the job lock (fd 3).
score_job(){  # <tag> <backbone> <stop k> <lane>
  local rc
  env $(job_env "$4" "$3") CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" \
    bash "$RUNNER" "$1" "$2" student "$HEAD_STEPS" >>"$RES/scores.log" 2>&1 \
    </dev/null
  rc=$?
  [ "$rc" -eq 0 ] || mark_failed "$1" "score rc=$rc"
  log "score $1 rc=$rc"
}

# Take the first job with no score, too few failures and a free lock. Runs
# its head, starts its score, and returns 0. Returns 1 when no job is left.
take_job(){  # <lane>
  local lane="$1" tier row code arm stop ckpt bytes tag rc
  while read -r tier row code arm stop ckpt bytes; do
    tag="$(tag_of "$arm" "$stop")"
    [ -s "$RES/score_$tag.txt" ] && continue
    [ "$(fail_count "$tag")" -lt "$TRIES" ] || continue
    exec 3>>"$LOCKS/$tag.lock"
    if ! flock -n 3; then exec 3>&-; continue; fi
    if [ -s "$RES/score_$tag.txt" ]; then exec 3>&-; continue; fi
    log "lane $lane: $code ${stop}k (tier $tier) -> $tag"
    # stdin is the job list of this loop, so the runner gets /dev/null.
    env $(job_env "$lane" "$stop") CF_SKIP_EVAL=1 \
      CF_BB_SHAPE="--d-model 384 --n-heads 8 --num-layers 3" \
      bash "$RUNNER" "$tag" "$CK/$ckpt" student "$HEAD_STEPS" \
      >>"$RES/heads.log" 2>&1 </dev/null
    rc=$?
    if [ "$rc" -ne 0 ]; then
      log "lane $lane: head $tag rc=$rc"
      mark_failed "$tag" "head rc=$rc"
      exec 3>&-
      return 0
    fi
    score_job "$tag" "$CK/$ckpt" "$stop" "$lane" &
    exec 3>&-
    return 0
  done < <(ordered_jobs)
  return 1
}

run_lane(){  # <lane>
  # The queue lock stays with the queue: a head or a score that outlives it
  # must not stop a new queue. The job locks keep each job single.
  exec 9>&-
  mkdir -p "/tmp/cf425_lane$1"
  while take_job "$1"; do :; done
  log "lane $1: no job left. Waiting for its scores."
  wait
  log "lane $1: done"
}

mkdir -p "$RES" "$LOCKS" "$FAILED" "$ROOT"
[ -f "$JOBS" ] || { log "ABORT: no job table at $JOBS"; exit 2; }
[ -f "$RUNNER" ] || { log "ABORT: no runner at $RUNNER"; exit 2; }
check_inputs || exit 3
if [ -n "${CF425_DRY_RUN:-}" ]; then
  ordered_jobs | while read -r tier row code arm stop ckpt bytes; do
    echo "$tier $code ${stop}k $(tag_of "$arm" "$stop")"
  done
  exit 0
fi

exec 9>>"$RES/queue.lock"
flock -n 9 || { log "ABORT: another queue holds $RES/queue.lock"; exit 4; }
log "start: $(ordered_jobs | wc -l) jobs, $LANES lanes, code $CODE ($(cat "$CODE/DEPLOYED_COMMIT" 2>/dev/null))"
for (( lane = 0; lane < LANES; lane++ )); do
  run_lane "$lane" &
  sleep "$STAGGER"
done
wait
log "QUEUE_END: $(ls "$RES"/score_*_recon.txt 2>/dev/null | wc -l) scores"
