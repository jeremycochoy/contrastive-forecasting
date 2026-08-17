#!/bin/bash
# #401 — elisa's role: every head, every 97-config GIFT-Eval, every figure.
#
# The study runs on four GPUs across two machines.
#
#   rented box, 2 GPUs   the two backbone arms, one per card
#                        (`scripts/launch_box.sh`). Backbones only: the eval
#                        reads gift-eval-data and the gift_eval package, and
#                        both live here.
#   elisa, GPU 0         this script. It trains one head per backbone stop as
#                        the stop arrives through the sync loop, runs that
#                        head's 97 GIFT-Eval configs on the CPU, and redraws
#                        the figures. GPU 1 belongs to another session; add
#                        it with HEAD_GPUS="0 1" once it frees.
#
# The three parts and their order:
#
#   1. the sync loop     `sync/launch_sync.sh <label>` — checked here, never
#                        started here, because it needs the box's ssh
#                        endpoint. CLAUDE.md: every remote run has one, for
#                        its whole duration.
#   2. heads_watch.sh    the heads and the evals, phase 1 then phase 2.
#   3. this loop         RUN_STATE.md and the figures, every FIGURE_EVERY
#                        seconds, so a session that picks this up reads one
#                        file and sees the current table.
#
# CF401_ROOT must be where the sync loop LANDS the box's checkpoints. The
# loop pulls the remote runs root into `<LOCAL_DIR>/sync`, keeping the
# relative tree, so that directory IS the root on this side.
#
# Everything is idempotent. Re-run this script after a reboot and every
# scored cell is a no-op.
#
# Usage:
#   CF401_ROOT=$HOME/cf401_sync/box_a/sync HEAD_GPUS="0" \
#     nohup setsid bash scripts/launch_elisa.sh &
#
#   CF401_DRY_RUN=1 bash scripts/launch_elisa.sh   # print the plan
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

HEAD_GPUS="${HEAD_GPUS:-0}"
FIGURE_EVERY="${FIGURE_EVERY:-1800}"
mkdir -p "$CF401_RESULTS" "$CF401_PLOTS"

STATE="$CF401_RESULTS/RUN_STATE.md"
LOG="$CF401_RESULTS/launch.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 elisa] $*" | tee -a "$LOG"; }

# What a re-dispatched session reads first. One file, overwritten, so it is
# never a log to scroll.
state(){  # <note>
  { echo "# #401 run state — the mean objective"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- note: $1"
    echo "- objective: \`--train-rollout-reduce $CF401_REDUCE\`, depths $CF401_DEPTHS"
    echo "- elisa pid: $$, head GPUs: $HEAD_GPUS"
    echo "- root (synced from the box): \`$CF401_ROOT\`"
    echo "- results: \`$CF401_RESULTS\`"
    echo
    echo "## Scores so far"
    echo
    echo '```'
    cat "$CF401_RESULTS/scores.csv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "## Backbone stops on this side"
    echo
    echo '```'
    ls -1 "$CF401_ROOT"/k*/*/leg_*k/*k.pth 2>/dev/null \
      | grep -v optimizer | sed "s#$CF401_ROOT/##" || echo "(none yet)"
    echo '```'
  } >"$STATE.tmp" && mv -f "$STATE.tmp" "$STATE"
}

# The sync loop is the only thing that puts the box's checkpoints here, so a
# missing loop is a study that never scores a cell. Verified by `ls` on the
# root, not by reading the loop's log (CLAUDE.md).
sync_check(){
  local n
  n="$(pgrep -fc "bash .*sync_loop.sh" 2>/dev/null || echo 0)"
  if [ "$n" -ge 1 ]; then
    log "sync: $n loop(s) running"
  else
    log "sync: NO sync_loop.sh runs. Start one before the box climbs:"
    log "  REMOTE_HOST=<ssh host> REMOTE_PORT=<port> \\"
    log "  REMOTE_DIR=/root/cf/reports/2026-08-15_rollout_depth_k16_8_32 \\"
    log "  REMOTE_RUNS=/root/cf401_runs LOCAL_DIR=$(dirname "$CF401_ROOT") \\"
    log "    bash sync/launch_sync.sh box_a"
  fi
}

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "elisa reduce=$CF401_REDUCE root=$CF401_ROOT results=$CF401_RESULTS"
  echo "  head gpus='$HEAD_GPUS' figures every ${FIGURE_EVERY}s"
  CF401_DRY_RUN=1 HEAD_GPUS="$HEAD_GPUS" bash "$HERE/heads_watch.sh"
  exit 0
fi

log "START reduce=$CF401_REDUCE depths='$CF401_DEPTHS' head gpus='$HEAD_GPUS'"
sync_check
state "starting"

HEAD_GPUS="$HEAD_GPUS" nohup bash "$HERE/heads_watch.sh" \
  >>"$CF401_RESULTS/heads_watch.out" 2>&1 &
watcher=$!
log "heads_watch pid $watcher"

# The watcher never returns on its own — the box keeps climbing, so there is
# always another stop coming. This loop is the reporting half: it redraws the
# figures and rewrites the state file while the watcher works, and it exits
# when the watcher does.
while kill -0 "$watcher" 2>/dev/null; do
  state "heads_watch pid $watcher"
  bash "$HERE/make_plots.sh" >>"$LOG" 2>&1
  sync_check
  sleep "$FIGURE_EVERY"
done
wait "$watcher"; rc=$?
bash "$HERE/make_plots.sh" 2>&1 | tee -a "$LOG"
log "heads_watch exited rc=$rc"
state "heads_watch exited rc=$rc"
exit $rc
