#!/bin/bash
# #373 round 3 — put every finished number into git, on a timer.
#
# The queue outlives the session that started it: the dispatcher, the
# supervisor, the budget guard and the sync loop all run detached, and the
# backbone column alone takes about 17 h. Collecting the evals by hand needs
# a session alive at the right moment, and there is no reason to think one
# will be. So this does it on a timer.
#
# Each tick:
#   1. r3_collect.sh   the flat round-3 tree -> the git checkout
#   2. the score files and the queue's own logs -> the git checkout
#   3. the coverage table -> results/coverage.txt, so every round is on disk
#   4. commit and push, only if something changed
#
# It never touches a file outside reports/2026-08-08_rollout_depth/, so a
# tick cannot pick up unrelated work in the checkout. It stops when the
# queue holds no job that is neither done nor failed.
#
# Usage: bash r3_publish.sh [interval seconds]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
GIT_ROOT="${GIT_ROOT:-/tmp/contrastive-forecasting-373}"
DST="$GIT_ROOT/reports/2026-08-08_rollout_depth"
REL="reports/2026-08-08_rollout_depth"
BRANCH="${BRANCH:-feature/contrastive-forecasting-373}"
INTERVAL="${1:-1200}"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [publish] $*" | tee -a "$RES/r3_publish.log"; }

left(){ # jobs neither done nor failed
  local n=0 id s
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF {print $1}' "$Q"); do
    s="$(cat "$STATE/$id.state" 2>/dev/null || echo queued)"
    case "$s" in done|failed) ;; *) n=$(( n + 1 ));; esac
  done; echo "$n"
}

log "start interval=${INTERVAL}s -> $DST on $BRANCH"
while :; do
  bash "$HERE/r3_collect.sh" "$GIT_ROOT" 2>&1 | tail -1 | tee -a "$RES/r3_publish.log"

  # The scores and the queue's own record. The coverage table reads the
  # score files, so they have to cross before it is regenerated.
  mkdir -p "$DST/results/queue"
  cp -f "$RES"/score_*.txt "$DST/results/" 2>/dev/null
  cp -f "$RES"/q_*.log "$RES"/execution_log.md "$DST/results/" 2>/dev/null
  cp -f "$RES"/queue/*.state "$RES"/queue/*.machine "$DST/results/queue/" 2>/dev/null
  cp -f "$RES"/r3_*.log "$DST/results/" 2>/dev/null

  # The coverage table, every round, on disk. `--md` is what the PR comment
  # and the report both take. It is written to the RUN checkout first: the
  # budget guard reads it from there when it posts its blocking comment, and
  # the guard runs whether or not a checkout is in sync.
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 300 python3 "$HERE/r2_coverage.py" >"$RES/coverage.txt" 2>&1
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 300 python3 "$HERE/r2_coverage.py" --md >"$RES/coverage.md" 2>&1
  cp -f "$RES/coverage.txt" "$RES/coverage.md" "$DST/results/" 2>/dev/null

  # The report's tables read the score files, so a new score moves them.
  timeout 600 python3 "$HERE/tables.py" --results "$DST/results" \
    --out "$DST/results/scores.md" --inject "$DST/rollout_depth.md" \
    >>"$RES/r3_publish.log" 2>&1
  timeout 600 python3 "$HERE/r2_tables.py" --results "$DST/results" \
    --out "$DST/results/scores_r2.md" >>"$RES/r3_publish.log" 2>&1

  if [ -n "$(git -C "$GIT_ROOT" status --porcelain -- "$REL")" ]; then
    git -C "$GIT_ROOT" add -A -- "$REL"
    git -C "$GIT_ROOT" commit -q -m "exp(#373): the round's numbers, as they land

$(grep -E '^deliverables' "$DST/results/coverage.txt" 2>/dev/null || true)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" \
      && log "committed $(git -C "$GIT_ROOT" rev-parse --short HEAD)"
    git -C "$GIT_ROOT" push -q origin "$BRANCH" 2>>"$RES/r3_publish.log" \
      && log "pushed" || log "push failed; will retry next tick"
  fi

  n="$(left)"
  log "queue has $n job(s) left"
  [ "$n" -eq 0 ] && { log "queue drained — publisher stands down"; exit 0; }
  sleep "$INTERVAL"
done
