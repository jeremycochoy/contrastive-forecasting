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

  # The scripts, too. The queue runs out of THIS tree, so a fix made while
  # the queue is live is made here, and until now nothing carried it into
  # git: three of them were edited on 2026-08-12 and only a hand copy would
  # have committed them. A run that cannot be reproduced from the branch is
  # not a reported run.
  cp -f "$HERE"/*.sh "$HERE"/*.py "$HERE"/*.tsv "$DST/scripts/" 2>/dev/null

  # The coverage table, every round, on disk. `--md` is what the PR comment
  # and the report both take. It is written to the RUN checkout first: the
  # budget guard reads it from there when it posts its blocking comment, and
  # the guard runs whether or not a checkout is in sync.
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 300 python3 "$HERE/r2_coverage.py" >"$RES/coverage.txt" 2>&1
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 300 python3 "$HERE/r2_coverage.py" --md >"$RES/coverage.md" 2>&1
  # The same grid by STAGE rather than by value. A table of numbers cannot
  # say what is still moving: a cell reads the same whether its number
  # landed this hour or last night. This is the one the round reprints.
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 300 python3 "$HERE/r2_coverage.py" --state >"$RES/coverage_state.txt" 2>&1
  cp -f "$RES/coverage.txt" "$RES/coverage.md" "$RES/coverage_state.txt" \
     "$DST/results/" 2>/dev/null

  # The same-arm pairs, again, on every tick. The card blocked publication
  # until each pair was shown to hold two models or one, and the fourth pair,
  # A2/B8, could not be tested: B8 had no checkpoint. It gains one at 40k and
  # another at 100k while this loop runs. `pair_identity.py` skips a stop
  # whose checkpoints are absent, so re-running it costs seconds and fills
  # the row the moment the file lands.
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 900 python3 "$HERE/pair_identity.py" --out "$RES/pair_identity.tsv" \
      >>"$RES/pair_identity.log" 2>&1
  cp -f "$RES/pair_identity.tsv" "$DST/results/" 2>/dev/null

  # ...and the FILES behind those pairs. `pair_identity.py` says whether two
  # cells hold the same weights; it cannot say whether they hold the same
  # FILE. Only the second question is the path bug the card blocked on, so
  # both tables ship, and both refresh when B8's checkpoints land.
  CF373_R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}" \
    timeout 900 python3 "$HERE/pair_head_files.py" --out "$RES/pair_head_files.tsv" \
      >>"$RES/pair_head_files.log" 2>&1
  cp -f "$RES/pair_head_files.tsv" "$RES/pair_head_files.log" "$DST/results/" 2>/dev/null

  # The stop ladder, the round's own figure. It reads the same score files
  # the tables do, so leaving it out of the tick would let the report show a
  # figure that disagrees with the table beside it. It draws in seconds and
  # needs no checkpoint, unlike rollout_fidelity and latent_movement, which
  # is why those two stay in make_report_assets.sh and this one does not.
  mkdir -p "$STUDY/plots" "$DST/plots"
  if timeout 300 python3 "$HERE/r2_plot_ladder.py" --results "$RES" \
       --out "$STUDY/plots/stop_ladder.png" >>"$RES/r3_publish.log" 2>&1; then
    cp -f "$STUDY/plots/stop_ladder.png" "$DST/plots/" 2>/dev/null
  else
    log "ladder plot failed; keeping the previous one"
  fi

  # The stop contrast with its interval. The ladder figure draws the levels;
  # this one draws bb200k minus bb100k and the bootstrap around it, which is
  # what the round's verdict rests on. The bootstrap is CPU-only and takes
  # about a minute for the whole set, so it re-runs on the tick and picks up
  # a cell the moment its second stop lands.
  timeout 1800 bash "$HERE/stop_bootstrap.sh" "$RES/stop_bootstrap.csv" \
    >>"$RES/stop_bootstrap.log" 2>&1
  cp -f "$RES/stop_bootstrap.csv" "$RES/stop_bootstrap.log" \
     "$DST/results/" 2>/dev/null
  if timeout 300 python3 "$HERE/plot_stop_delta.py" --results "$RES" \
       --out "$STUDY/plots/stop_delta.png" >>"$RES/r3_publish.log" 2>&1; then
    cp -f "$STUDY/plots/stop_delta.png" "$DST/plots/" 2>/dev/null
  else
    log "stop-delta plot failed; keeping the previous one"
  fi

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
