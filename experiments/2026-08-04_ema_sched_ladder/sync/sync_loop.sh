#!/bin/bash
# #393 persistent sync loop — 15-min ticks pulling the ladder's artefacts
# from a training host into a permanent local checkout. CLAUDE.md § Remote
# Machine Monitoring: EVERY remote run has a sync loop for its full
# duration, atomic writes only, per-class size floors, ls-verify not
# log-verify.
#
# The ladder writes in two places and both have to come back:
#
#   $REMOTE_RUNS/<cell>/leg_<N>k/  backbone + optimizer state + losses CSV,
#                                  one dir per stop
#   $REMOTE_RUNS/<cell>/eval/      quantile heads, their encoder-source
#                                  markers, GIFT-Eval outputs, score files
#   $REMOTE_DIR/results/           ladder.csv, decisions.csv, run logs
#
# A cell's stop list is not known in advance — the extend rule decides how
# far it goes — so each tick ASKS the remote what exists rather than
# guessing filenames. One ssh per cell, then one pull per file whose local
# size does not already match. Enumerating every possible step instead
# would mean thousands of scp attempts per tick against files that are not
# there, and a tick has to fit in the interval.
#
# Usage on the machine that OWNS the local persistent checkout:
#   REMOTE_HOST=elisa SSH_USER=jupyter \
#   REMOTE_DIR=~/workspaces/contrastive-forecasting/experiments/2026-08-04_ema_sched_ladder \
#   REMOTE_RUNS=/home/jupyter/checkpoints_backup/cf-393 \
#   LOCAL_DIR=/absolute/path/to/experiments/2026-08-04_ema_sched_ladder \
#     nohup setsid bash sync_loop.sh > sync_loop.log 2>&1 &
#
# Verify by `ls`, not by reading this loop's log: a missing failure line
# can mean the pattern never matched.
set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:?REMOTE_HOST must be set (e.g. elisa or ssh5.vast.ai)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:?REMOTE_DIR must be set (experiment dir on REMOTE_HOST)}"
REMOTE_RUNS="${REMOTE_RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
LOCAL_DIR="${LOCAL_DIR:?LOCAL_DIR must be set (absolute path in the persistent checkout)}"
INTERVAL="${INTERVAL:-900}"        # 15-min ticks (CLAUDE.md)
export SSH_USER="${SSH_USER:-root}"   # vast.ai is root; elisa is jupyter

case "$LOCAL_DIR" in
  /tmp|/tmp/*)
    echo "ABORT: LOCAL_DIR=$LOCAL_DIR is under /tmp — refusing to sync into" >&2
    echo "  an ephemeral path." >&2
    exit 2 ;;
esac

# safe_pull.sh: atomic scp with .tmp → mv, .prev rotation, per-file size
# floor. Raw scp writes straight to the destination, so a mid-transfer drop
# corrupts the prior good copy.
SAFE_PULL="$(dirname "$LOCAL_DIR")/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
[ -f "$SAFE_PULL" ] || { echo "ABORT: safe_pull.sh not at $SAFE_PULL" >&2; exit 2; }

CELLS=(arm6_v2_combab_alignS arm6_v2_combab_alignT
       arm5_combab_alignS arm5_combab_alignT
       arm6_v2_ncpc_alignS arm6_v2_ncpc_alignT
       arm6_v2_nse_alignS arm6_v2_nse_alignT
       arm4_combab arm1_nse)

# Per-class floors, never one blanket number, and every one of them MEASURED
# against what this run produces rather than guessed:
#
#   backbone           5,192,927 B    floor 3 MB
#   backbone optimizer 5,789,873 B    floor 4 MB
#   quantile head        449,267 B    floor 200 kB
#   head optimizer       902,143 B    floor 400 kB
#
# Both guessed floors were wrong, and wrong in the direction that loses
# data. The optimizer floor was 6 MB against a real 5.79 MB and dropped
# every backbone optimizer on the first tick; the head floor was 1 MB
# against a real 449 kB. That is PR #45's failure mode — a wrong-but-large
# floor discards a good file and logs one `✗` line. The cost is not
# symmetric: a resumed leg without its optimizer companion loses the step
# counter, the RNG state and AdamW's moments, and a stop without its head
# has no number. Re-measure whenever the architecture or the head arch
# changes. CSVs, logs, markers and score files are a few bytes upward.
backbone_floor(){ echo 3000000; }
optimizer_floor(){ echo 4000000; }
head_floor(){ echo 200000; }
head_optimizer_floor(){ echo 400000; }
text_floor(){ echo 1; }

# The floor a remote path falls under, by name. Order matters: a head's
# optimizer companion is named `qhead_..._optimizer.pth` and so matches
# BOTH the head pattern and the optimizer pattern. It has to be tested
# first, or a 902 kB file is measured against the 4 MB backbone-optimizer
# floor and thrown away. Unknown names get the text floor rather than a
# checkpoint floor, for the same reason.
floor_for(){
  case "$1" in
    qhead_*_optimizer.pth) head_optimizer_floor ;;
    qhead_*.pth)           head_floor ;;
    *_optimizer.pth)       optimizer_floor ;;
    *.pth)                 backbone_floor ;;
    *)                     text_floor ;;
  esac
}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sync-393] $*"; }

# `<size> <path>` for every file under a remote directory, or nothing.
remote_listing(){  # <remote dir>
  ssh -o StrictHostKeyChecking=no -p "$REMOTE_PORT" \
      "${SSH_USER}@${REMOTE_HOST}" \
      "find '$1' -type f -printf '%s %p\n' 2>/dev/null" 2>/dev/null
}

pull_tree(){  # <remote dir> <local dir>
  local rdir="$1" ldir="$2" size path rel dst
  while read -r size path; do
    [ -n "${path:-}" ] || continue
    rel="${path#$rdir/}"
    dst="$ldir/$rel"
    # Already here at the same size: nothing changed, skip the transfer.
    [ -f "$dst" ] && [ "$(wc -c <"$dst")" = "$size" ] && continue
    mkdir -p "$(dirname "$dst")"
    bash "$SAFE_PULL" "$REMOTE_HOST" "$REMOTE_PORT" "$path" "$dst" \
         "$(floor_for "$(basename "$path")")" \
      || echo "  (skip: $path did not land)"
  done < <(remote_listing "$rdir")
}

log "start REMOTE=${SSH_USER}@$REMOTE_HOST:$REMOTE_DIR RUNS=$REMOTE_RUNS"
log "      LOCAL=$LOCAL_DIR interval=${INTERVAL}s"
mkdir -p "$LOCAL_DIR/sync" "$LOCAL_DIR/results"

while true; do
  log "tick"
  for cell in "${CELLS[@]}"; do
    pull_tree "$REMOTE_RUNS/$cell" "$LOCAL_DIR/sync/$cell"
  done
  # The ladder's own record: which stops ran, what they scored, which
  # branch of the extend rule fired, and the per-cell logs.
  pull_tree "$REMOTE_DIR/results" "$LOCAL_DIR/results"
  log "--- sleeping ${INTERVAL}s ---"
  sleep "$INTERVAL"
done
