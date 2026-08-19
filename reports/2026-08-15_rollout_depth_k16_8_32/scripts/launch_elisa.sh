#!/bin/bash
# #401 — elisa's role: every head, every 97-config GIFT-Eval, every figure.
#
# The study runs on four GPUs across two machines. This is the layout, and it
# is not a choice left to the reader of a launch command.
#
#   rented box, 2 GPUs   the two backbone arms, k = 8 and k = 32, one per card
#                        (`scripts/launch_box.sh`). Backbones only: the eval
#                        reads gift-eval-data and the gift_eval package, and
#                        both live here. The box's root is the SINGLE source
#                        for backbones.
#   elisa, GPU 0         this script. It trains one head per backbone stop as
#                        the stop arrives through the sync loop, runs that
#                        head's 97 GIFT-Eval configs on the CPU, and redraws
#                        the figures. GPU 1 belongs to another session and
#                        this study does not take it.
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
# The root is CF401_SYNC_ROOT, where the sync loop LANDS the box's tree. The
# loop pulls the remote runs root into `<CF401_SYNC_DIR>/sync` keeping the
# relative paths, so that directory IS the root on this side. It is never a
# local checkpoints directory: a backbone under one root and a watcher on
# another is an arm that climbs for 33 h and is never scored, which
# `heads_watch.sh` refuses out loud.
#
# Everything is idempotent. Re-run this script after a reboot and every
# scored cell is a no-op.
#
# Usage:
#   HEAD_GPUS="0" nohup setsid bash scripts/launch_elisa.sh &
#
#   CF401_ROOT=<other tree> bash scripts/launch_elisa.sh   # read another tree
#   CF401_DRY_RUN=1 bash scripts/launch_elisa.sh           # print the plan
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF401_ROOT_GIVEN="${CF401_ROOT:-}"
. "$HERE/study.sh"
cf401_use_root "$CF401_SYNC_ROOT"

HEAD_GPUS="${HEAD_GPUS:-0}"
FIGURE_EVERY="${FIGURE_EVERY:-1800}"
mkdir -p "$CF401_RESULTS" "$CF401_PLOTS"

LOG="$CF401_RESULTS/launch.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 elisa] $*" | tee -a "$LOG"; }

# What a re-dispatched session reads first. `run_state.sh` builds it, so the
# file is not owned by this launcher: any session can rebuild it at any
# moment, with no launcher running. That matters, because this launcher is
# not the only driver a multi-day study has — and while it was the only
# writer, the file froze at "(none yet)" the moment the launcher exited.
state(){  # <note>
  bash "$HERE/run_state.sh" "$1 (elisa pid $$, head GPUs $HEAD_GPUS)" \
    >/dev/null
}

# The sync loop is the only thing that puts the box's checkpoints here, so a
# missing loop is a study that never scores a cell.
#
# The loop this study needs is the one whose LOCAL ROOT is this study's, and
# `cf401_sync_loops` identifies a loop by its working directory. elisa is
# shared: a count over every `sync_loop.sh` on the machine reported another
# study's pull as this one's.
SYNC_LOCAL="${SYNC_LOCAL:-$CF401_SYNC_DIR}"

sync_check(){
  local n
  n="$(cf401_sync_loops "$SYNC_LOCAL")"
  if [ "$n" -ge 1 ]; then
    log "sync: $n loop(s) running for $SYNC_LOCAL"
  else
    log "sync: NO sync_loop.sh runs for $SYNC_LOCAL. Start one before the box climbs:"
    log "  REMOTE_HOST=<ssh host> REMOTE_PORT=<port> \\"
    log "  REMOTE_DIR=/root/cf/reports/2026-08-15_rollout_depth_k16_8_32 \\"
    log "  LOCAL_DIR=$SYNC_LOCAL \\"
    log "    bash sync/launch_sync.sh $CF401_BOX_LABEL"
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
