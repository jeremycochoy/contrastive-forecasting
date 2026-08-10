#!/bin/bash
# #373 round 2 — turn synced heads into GM-Relative MASE, on elisa's cores.
#
# Usage: bash r2_eval_driver.sh [poll seconds]
#
# The boxes train the backbone and both heads. This loop watches what their
# sync loops bring back, pairs each head with the backbone that produced it,
# and runs the 97-config GIFT-Eval here. One machine produces every number
# in this study, so no rented card can put one on a different scale.
#
# Idempotent at every level: eval_local.sh skips a tag whose score file
# exists, and the official eval resumes per config, so a restarted driver
# re-runs nothing that finished.
set -uo pipefail

POLL="${1:-180}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
export WT="${WT:-/home/jupyter/wt-cf-373-run2}"
SYNC_BASE="${CF373_R2_SYNC:-/home/jupyter/cf373_r2}"
HEAD_SEED="${HEAD_SEED:-20260722}"
export CF393_EVAL_SLOTS="${CF393_EVAL_SLOTS:-6}"
export EVAL_SHARDS="${EVAL_SHARDS:-4}"
export GIFT_EVAL="${GIFT_EVAL:-/home/jupyter/workspaces/gift-eval-data}"
mkdir -p "$RES"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [evald] $*" | tee -a "$RES/r2_eval_driver.log"; }

log "start poll=${POLL}s slots=$CF393_EVAL_SLOTS shards=$EVAL_SHARDS base=$SYNC_BASE"
declare -A started=()

while :; do
  for headck in "$SYNC_BASE"/*/sync/eval/*/qhead_*_s${HEAD_SEED}_final.pth; do
    [ -f "$headck" ] || continue
    outdir="$(dirname "$headck")"
    tag="$(basename "$outdir")"                        # <cell>_k<K>_bb<N>k_<enc>
    celldir="${headck%%/sync/eval/*}"
    cell="${tag%%_*}"
    enc="${tag##*_}"
    stopk="$(sed -E 's/.*_bb([0-9]+)k_.*/\1/' <<<"$tag")"
    kk="$(sed -E 's/.*_k([0-9]+)_bb.*/\1/' <<<"$tag")"
    score="$RES/score_${tag}.txt"
    [ -s "$score" ] && continue
    [ -n "${started[$tag]:-}" ] && continue

    # The backbone the head was trained on, in this cell's own sync tree.
    bb="$(CF373_ROOT="$celldir/sync" bash -c \
          ". '$HERE/cell_paths.sh'; cf373_bb_ckpt '$cell' '$kk' '$(( stopk * 1000 ))'")"
    if [ -z "$bb" ] || [ ! -f "$bb" ]; then
      log "WAIT $tag — head is here, its bb${stopk}k checkpoint is not yet"
      continue
    fi
    # The head records which checkpoint file it read. A mismatch means the
    # sync brought back a different backbone than the one the head saw, and
    # the pair would be scored as if it were the cell's.
    if [ -f "$outdir/backbone.txt" ]; then
      want="$(cat "$outdir/backbone.txt")"
      [ "$(basename "$bb")" = "$want" ] || {
        log "SKIP $tag — head was trained on $want, local bb is $(basename "$bb")"
        continue; }
    fi

    log "EVAL $tag  bb=$(basename "$bb")"
    started[$tag]=1
    ( bash "$HERE/eval_local.sh" "$tag" "$stopk" "$enc" "$bb" "$headck" \
        "$outdir" "$score" >>"$RES/r2_eval_${tag}.log" 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "[$(date '+%m-%d %H:%M:%S')] [evald] DONE $tag $(cat "$score")" \
          | tee -a "$RES/r2_eval_driver.log"
      else
        echo "[$(date '+%m-%d %H:%M:%S')] [evald] FAIL $tag rc=$rc" \
          | tee -a "$RES/r2_eval_driver.log"
      fi ) &
  done

  # A tag that failed is retried on the next sweep that sees no score for
  # it, so clear the guard for anything no longer running.
  for t in "${!started[@]}"; do
    [ -s "$RES/score_${t}.txt" ] && continue
    pgrep -f "eval_local.sh $t " >/dev/null 2>&1 || unset "started[$t]"
  done

  sleep "$POLL"
done
