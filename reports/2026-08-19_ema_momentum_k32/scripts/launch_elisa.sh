#!/bin/bash
# #404 — elisa's role: every head, every 97-config GIFT-Eval, every figure.
#
# The study runs on three GPUs across two machines.
#
#   rented box, 2 GPUs   the four backbone arms, two per card
#                        (`scripts/launch_box.sh`). Backbones only: the eval
#                        reads gift-eval-data and the `gift_eval` package, and
#                        both live here. The box's root is the SINGLE source
#                        for backbones.
#   elisa, GPU 0         this script. It trains one head per arm as the arm
#                        arrives through the sync loop, runs that head's 97
#                        GIFT-Eval configs on the CPU, and redraws the four
#                        deliverables. GPU 1 belongs to another session.
#
# The three parts and their order:
#
#   1. the sync loop     `sync/launch_sync.sh <label>` — checked here, never
#                        started here, because it needs the box's ssh
#                        endpoint. CLAUDE.md: every remote run has one, for
#                        its whole duration.
#   2. heads_watch.sh    the heads and the evals.
#   3. this loop         RUN_STATE.md and the figures, every FIGURE_EVERY
#                        seconds, so a session that picks this up reads one
#                        file and sees the current table.
#
# Everything is idempotent. Re-run this after a reboot and every scored arm is
# a no-op.
#
# Usage:
#   HEAD_GPUS="0" nohup setsid bash scripts/launch_elisa.sh &
#
#   CF404_ROOT=<other tree> bash scripts/launch_elisa.sh   # read another tree
#   CF404_DRY_RUN=1 bash scripts/launch_elisa.sh           # print the plan
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF404_ROOT_GIVEN="${CF404_ROOT:-}"
. "$HERE/study.sh"
cf404_use_root "$CF404_SYNC_ROOT"

HEAD_GPUS="${HEAD_GPUS:-0}"
FIGURE_EVERY="${FIGURE_EVERY:-1800}"
mkdir -p "$CF404_RESULTS" "$CF404_PLOTS"

STATE="$CF404_RESULTS/RUN_STATE.md"
LOG="$CF404_RESULTS/launch.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#404 elisa] $*" | tee -a "$LOG"; }

# What a re-dispatched session reads first. One file, overwritten, so it is
# never a log to scroll.
state(){  # <note>
  { echo "# #404 run state — the EMA momentum sweep at k = 32"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- note: $1"
    echo "- cell: \`$CF404_CELL\`, k = $CF404_K, reduce \`$CF404_REDUCE\`"
    echo "- arms: $CF404_ARMS"
    echo "- elisa pid: $$, head GPUs: $HEAD_GPUS"
    echo "- root (synced from the box): \`$CF404_ROOT\`"
    echo "- results: \`$CF404_RESULTS\`"
    echo
    echo "## Scores so far"
    echo
    echo '```'
    cat "$CF404_RESULTS/scores.csv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "## Backbone stops on this side"
    echo
    echo '```'
    ls -1 "$CF404_ROOT"/*/*/leg_*k/*k.pth 2>/dev/null \
      | grep -v optimizer | sed "s#$CF404_ROOT/##" || echo "(none yet)"
    echo '```'
  } >"$STATE.tmp" && mv -f "$STATE.tmp" "$STATE"
}

# The sync loop is the only thing that puts the box's checkpoints here, so a
# missing loop is a study that never scores an arm.
#
# The loop this study needs is the one whose LOCAL ROOT is this study's, and
# `cf404_sync_loops` identifies a loop by its working directory. elisa is
# shared: a count over every `sync_loop.sh` on the machine would report another
# study's pull as this one's.
SYNC_LOCAL="${SYNC_LOCAL:-$CF404_SYNC_DIR}"

sync_check(){
  local n
  n="$(cf404_sync_loops "$SYNC_LOCAL")"
  if [ "$n" -ge 1 ]; then
    log "sync: $n loop(s) running for $SYNC_LOCAL"
  else
    log "sync: NO sync_loop.sh runs for $SYNC_LOCAL. Start one before the box climbs:"
    log "  REMOTE_HOST=<ssh host> REMOTE_PORT=<port> \\"
    log "  REMOTE_DIR=/root/cf/reports/2026-08-19_ema_momentum_k32 \\"
    log "  LOCAL_DIR=$SYNC_LOCAL \\"
    log "    bash sync/launch_sync.sh $CF404_BOX_LABEL"
  fi
}

if [ -n "${CF404_DRY_RUN:-}" ]; then
  echo "elisa root=$CF404_ROOT results=$CF404_RESULTS"
  echo "  head gpus='$HEAD_GPUS' figures every ${FIGURE_EVERY}s"
  CF404_DRY_RUN=1 HEAD_GPUS="$HEAD_GPUS" bash "$HERE/heads_watch.sh"
  exit 0
fi

log "START arms='$CF404_ARMS' head gpus='$HEAD_GPUS'"
sync_check
state "starting"

HEAD_GPUS="$HEAD_GPUS" nohup bash "$HERE/heads_watch.sh" \
  >>"$CF404_RESULTS/heads_watch.out" 2>&1 &
watcher=$!
log "heads_watch pid $watcher"

# The watcher returns when every arm on this side is scored. This loop is the
# reporting half: it redraws the figures and rewrites the state file while the
# watcher works, and it exits when the watcher does.
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
