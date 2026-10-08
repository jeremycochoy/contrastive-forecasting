#!/bin/bash
# #425 — the R score of another snapshot of the head of one job.
#
# The queue scores the final head of each job (step 30,000). The head
# trainer also keeps the head of its best training loss (`*_best.pth`), and
# for some jobs that head is from an earlier step. This script scores such
# a snapshot with the eval of the queue. The two R scores of one head
# training then show how much R moves between two snapshots of one head.
# `final` scores the final head again, as a control of this path.
#
# The snapshot gets its own tag (`<arm>_bb<stop>k_h30k_<snapshot>_recon`),
# its own head folder and its own results folder. So the queue does not see
# it. The script trains no head: it reads a head that elisa holds.
#
# On the box (the default), the score takes one of the eval slots of the
# queue, so the box runs no more scores at the same time than the queue
# plans.
#
# On elisa (CF425_SNAP_GPU=<N>), the score runs on GPU <N> of elisa, and the
# box is not used. Every file is under ~/checkpoints_backup/cf-425-snap, so a
# restart of elisa keeps it:
#   code/     the code that the score reads. Fill it one time, and after each
#             code change:
#             CF425_ELISA_BASE=~/checkpoints_backup/cf-425-snap bash deploy_elisa.sh
#   ckpt/     the copy of the head, and the files of its eval.
#   results/  the score, the logs and the eval slots (3 for each GPU).
# The eval reads the checkpoint of the run in elisa's mirror of the box.
#
# Usage, on elisa:
#   bash snapshot_score.sh <code> <stop k> <best|final>                     # on the box
#   CF425_SNAP_GPU=<N> bash snapshot_score.sh <code> <stop k> <best|final>  # on elisa
# The score of the box comes to elisa with the next tick of sync_box.sh:
#   <results mirror>/snapshots/score_<tag>.txt
# The score of elisa:  ~/checkpoints_backup/cf-425-snap/results/score_<tag>.txt
set -uo pipefail

RUN="${1:?usage: snapshot_score.sh <code> <stop k> <best|final>}"
STOP="${2:?stop in thousands}"
SNAP="${3:?best|final}"
case "$SNAP" in best|final) ;; *) echo "ABORT: snapshot '$SNAP'. Use best or final." >&2; exit 2 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${CF425_JOBS:-$HERE/jobs.tsv}"
HOST="${CF425_HOST:-root@ssh5.vast.ai}"
PORT="${CF425_PORT:-31200}"
BACKUP="${CF425_MIRROR:-$HOME/checkpoints_backup/cf-412/vast_lr100x}"
MIRROR="$BACKUP/cf-425"
ELISA_GPU="${CF425_SNAP_GPU:-}"
SEED=20260722
SHAPE="--d-model 384 --n-heads 8 --num-layers 3"
if [ -n "$ELISA_GPU" ]; then
  [[ "$ELISA_GPU" =~ ^[0-9]+$ ]] || { echo "ABORT: CF425_SNAP_GPU='$ELISA_GPU' is not a GPU number." >&2; exit 2; }
  BASE="${CF425_SNAP_BASE:-$HOME/checkpoints_backup/cf-425-snap}"
  CODE="${CF425_CODE:-$BASE/code}"
  CK="${CF425_CK:-$BACKUP}"
  ROOT="$BASE/ckpt"
  RES="${CF425_RES:-$BASE/results}"
else
  CODE="${CF425_CODE:-/workspace/cf-425}"
  CK="${CF425_CK:-/workspace/ckpt}"
  ROOT="$CK/cf-425/snapshots"
  RES="${CF425_RES:-/workspace/results/cf-425}/snapshots"
fi
RUNNER="${CF425_RUNNER:-$CODE/reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh}"

read -r arm ckpt < <(awk -F'\t' -v c="$RUN" -v s="$STOP" \
  '!/^#/ && $1 == c && $3 == s { print $2, $5 }' "$JOBS")
[ -n "${arm:-}" ] || { echo "ABORT: no job $RUN ${STOP}k in $JOBS" >&2; exit 2; }
job="${arm}_bb${STOP}k_h30k_recon"
tag="${arm}_bb${STOP}k_h30k_${SNAP}_recon"
head="$MIRROR/recon/eval/$job/qhead_${job}_s${SEED}_${SNAP}.pth"
[ -s "$head" ] || { echo "ABORT: no head at $head" >&2; exit 3; }
bytes=$(stat -c %s "$head")
out="$ROOT/eval/$tag"
dest="$out/qhead_${tag}_s${SEED}_final.pth"

if [ -n "$ELISA_GPU" ]; then
  [ -f "$RUNNER" ] || { echo "ABORT: no code at $CODE. Run deploy_elisa.sh with CF425_ELISA_BASE=$BASE." >&2; exit 2; }
  [ -f "$CK/$ckpt" ] || { echo "ABORT: no checkpoint at $CK/$ckpt" >&2; exit 3; }
  # The head takes its name only at its full size, as on the box.
  mkdir -p "$out" "$RES" || exit 4
  cp "$head" "$dest.tmp" && [ "$(stat -c %s "$dest.tmp")" = "$bytes" ] \
    && mv -f "$dest.tmp" "$dest" \
    || { echo "ABORT: the copy of $head is not $bytes bytes" >&2; exit 4; }
  # The environment of a score of queue.sh (job_env), with the folders of
  # the snapshots of elisa. Each GPU has its own eval slots.
  nohup setsid env WT="$CODE" CF373_ROOT="$ROOT" CF_RESULTS="$RES" \
    CF_STOP_K="$STOP" GIFT_EVAL="${GIFT_EVAL:-$HOME/workspaces/gift-eval-data}" \
    EVAL_SHARDS=2 EVAL_DEVICE=cuda CF393_EVAL_SLOTS="${CF425_SNAP_SLOTS:-3}" \
    CF393_EVAL_SLOTDIR="$RES/evalslots_gpu$ELISA_GPU" BB_GPU="$ELISA_GPU" \
    CF_RECONSTRUCTION=encoder CF_BB_SHAPE="$SHAPE" \
    bash "$RUNNER" "$tag" "$CK/$ckpt" student 30000 \
    >>"$RES/run.log" 2>&1 </dev/null &
  echo "started $tag on GPU $ELISA_GPU of elisa ($bytes bytes): $RES/score_$tag.txt"
  exit 0
fi

# The head goes to the box under a temporary name, and takes its name only
# at its full size.
ssh -p "$PORT" "$HOST" "mkdir -p '$out' '$RES'" </dev/null || exit 4
scp -q -P "$PORT" "$head" "$HOST:$dest.tmp" || exit 4
ssh -p "$PORT" "$HOST" "[ \"\$(stat -c %s '$dest.tmp')\" = $bytes ] && mv -f '$dest.tmp' '$dest'" </dev/null \
  || { echo "ABORT: the box copy of $head is not $bytes bytes" >&2; exit 4; }

# The environment of a score of queue.sh (job_env), with the folders of the
# snapshots.
ssh -p "$PORT" "$HOST" "nohup setsid env WT='$CODE' CF373_ROOT='$ROOT' \
  CF_RESULTS='$RES' CF_STOP_K='$STOP' GIFT_EVAL=/workspace/gift-eval-data \
  EVAL_SHARDS=2 EVAL_DEVICE=cuda CF393_EVAL_SLOTS=2 \
  CF393_EVAL_SLOTDIR=/tmp/cf425_evalslots BB_GPU=0 CF_RECONSTRUCTION=encoder \
  CF_BB_SHAPE='$SHAPE' \
  bash '$RUNNER' \
  '$tag' '$CK/$ckpt' student 30000 >>'$RES/run.log' 2>&1 </dev/null &" </dev/null
echo "started $tag on the box ($bytes bytes): $RES/score_$tag.txt"
