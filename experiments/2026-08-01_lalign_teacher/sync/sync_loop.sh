#!/bin/bash
# #390 persistent sync loop — 15-min ticks pulling backbone checkpoints,
# losses CSVs and training logs off the training host into a permanent
# checkout. Written to satisfy CLAUDE.md § Remote Machine Monitoring: EVERY
# remote run has a sync loop for its full duration, atomic writes only,
# per-class size thresholds, verify by `ls` and not by reading this log.
#
# Companion to scripts/monitor.sh. monitor.sh runs ON the training host and
# guards the small CSVs there; this loop runs on the machine that owns the
# durable copy and pulls everything, checkpoints included.
#
# The sweep is 10 cells x 3 waves (see scripts/orchestrate.sh). Each arm
# writes $REMOTE_DIR/runs/${NAME}_* and $REMOTE_DIR/results/*; the loop
# iterates the 10 arm names and pulls whatever is there. It never blocks on
# a missing file (safe_pull.sh returns non-zero and leaves the local copy
# untouched).
#
# Usage on the machine that OWNS the local persistent checkout:
#   REMOTE_HOST=elisa \
#   REMOTE_DIR=~/workspaces/contrastive-forecasting/experiments/2026-08-01_lalign_teacher \
#   LOCAL_DIR=/absolute/path/to/experiments/2026-08-01_lalign_teacher \
#     nohup setsid bash sync_loop.sh > sync_loop.log 2>&1 &
set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:?REMOTE_HOST must be set (e.g. elisa or ssh5.vast.ai)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:?REMOTE_DIR must be set (path on REMOTE_HOST)}"
LOCAL_DIR="${LOCAL_DIR:?LOCAL_DIR must be set (absolute path in the persistent checkout)}"
INTERVAL="${INTERVAL:-900}"           # 15-min ticks (CLAUDE.md)
FINAL_SENTINEL="${FINAL_SENTINEL:-}"  # optional single-arm early exit

case "$LOCAL_DIR" in
  /tmp/*|/tmp)
    echo "ABORT: LOCAL_DIR=$LOCAL_DIR is under /tmp — refusing to sync into an ephemeral path." >&2
    exit 2
    ;;
esac

# safe_pull.sh: atomic scp with .tmp → mv, .prev rotation, per-file size
# floor. Lives in the periodic-synth-mix scripts dir (added PR #45). Raw scp
# writes straight to the destination, so a mid-transfer drop would corrupt
# the previous good copy.
SAFE_PULL="$(dirname "$LOCAL_DIR")/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
[ -f "$SAFE_PULL" ] || {
  echo "ABORT: safe_pull.sh not at $SAFE_PULL" >&2
  exit 2
}

# Run names are derived, not re-typed — a hand-copied table is how a sync
# loop silently stops pulling an arm. bb_name() is pinned against
# run_arm.sh's case block by tests/test_390_launcher_shape.py.
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../scripts/arm_names.sh
source "$HERE/../scripts/arm_names.sh"
ARMS_STR="${ARMS:-${CF390_ARMS[*]}}"
read -r -a ARMS <<< "$ARMS_STR"

# Per-class minimum-byte floors (never one blanket number — CLAUDE.md: a
# blanket 70 MB floor on *.pth silently dropped 2.4 MB head checkpoints in
# PR #45). Small backbone (d_model=64, 3+3 layers) → ~5 MB checkpoint and
# ~10 MB optimizer state; CSVs and logs a few KB.
BACKBONE_MIN=3000000         # ~5 MB actual, floor 3 MB
BACKBONE_OPT_MIN=6000000     # ~10 MB actual, floor 6 MB
TEXT_MIN=100                 # CSVs / logs: at least a header

# Checkpoint step_k values the three waves put on disk, derived from the wave
# table in scripts/arm_names.sh rather than retyped here: a hand-kept list
# stops pulling the snapshot the next wave resumes from the moment a cadence
# moves. safe_pull silently skips missing files, so listing every possible
# step is cheap — a wave-1-only run simply no-ops on the later entries.
BACKBONE_STEPS_K="${BACKBONE_STEPS_K:-$(wave_checkpoint_steps_k | tr '\n' ' ')}"

pull(){ # remote_path local_path min_bytes
  bash "$SAFE_PULL" "$REMOTE_HOST" "$REMOTE_PORT" "$1" "$2" "$3" \
    || echo "  (skip: $1 not present or below floor)"
}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sync-390] $*"; }

log "start REMOTE=$REMOTE_HOST:$REMOTE_DIR LOCAL=$LOCAL_DIR interval=${INTERVAL}s arms=${#ARMS[@]}"
mkdir -p "$LOCAL_DIR/runs" "$LOCAL_DIR/results"

while true; do
  log "tick"
  for arm in "${ARMS[@]}"; do
    NAME="$(bb_name "$arm")" || { log "  (skip: unknown arm $arm)"; continue; }
    # Best-checkpoint pairs (train.py writes on each new best-loss / best-gap).
    # ALWAYS pull the optimizer beside the backbone — without it a resume
    # loses the step counter, the RNG state and AdamW momentum.
    for f in best_gap.pth best_loss.pth; do
      pull "$REMOTE_DIR/runs/${NAME}_$f" \
           "$LOCAL_DIR/runs/${NAME}_$f" "$BACKBONE_MIN"
      pull "$REMOTE_DIR/runs/${NAME}_${f%.pth}_optimizer.pth" \
           "$LOCAL_DIR/runs/${NAME}_${f%.pth}_optimizer.pth" "$BACKBONE_OPT_MIN"
    done
    # Training-dynamics CSVs and the training log — the irreplaceable part.
    pull "$REMOTE_DIR/runs/${NAME}_losses.csv"         "$LOCAL_DIR/runs/${NAME}_losses.csv"         "$TEXT_MIN"
    pull "$REMOTE_DIR/runs/${NAME}_attn_amplitude.csv" "$LOCAL_DIR/runs/${NAME}_attn_amplitude.csv" "$TEXT_MIN"
    pull "$REMOTE_DIR/results/run_${NAME}.log"         "$LOCAL_DIR/results/run_${NAME}.log"         "$TEXT_MIN"
    # Periodic step snapshots (backbone + optimizer). These are what the
    # next wave resumes from, so missing one costs a whole wave.
    for sk in $BACKBONE_STEPS_K; do
      pull "$REMOTE_DIR/runs/${NAME}_${sk}k.pth"           "$LOCAL_DIR/runs/${NAME}_${sk}k.pth"           "$BACKBONE_MIN"
      pull "$REMOTE_DIR/runs/${NAME}_${sk}k_optimizer.pth" "$LOCAL_DIR/runs/${NAME}_${sk}k_optimizer.pth" "$BACKBONE_OPT_MIN"
    done
    # Restart-suffixed variants (safe_run_name adds _r2, _r3, …).
    for rk in _r2 _r3; do
      pull "$REMOTE_DIR/runs/${NAME}${rk}_final.pth"           "$LOCAL_DIR/runs/${NAME}${rk}_final.pth"           "$BACKBONE_MIN"
      pull "$REMOTE_DIR/runs/${NAME}${rk}_final_optimizer.pth" "$LOCAL_DIR/runs/${NAME}${rk}_final_optimizer.pth" "$BACKBONE_OPT_MIN"
      pull "$REMOTE_DIR/runs/${NAME}${rk}_losses.csv"          "$LOCAL_DIR/runs/${NAME}${rk}_losses.csv"          "$TEXT_MIN"
    done
    # Final backbone snapshot + the launcher-copied FINAL sentinel.
    pull "$REMOTE_DIR/runs/${NAME}_final.pth"           "$LOCAL_DIR/runs/${NAME}_final.pth"           "$BACKBONE_MIN"
    pull "$REMOTE_DIR/runs/${NAME}_final_optimizer.pth" "$LOCAL_DIR/runs/${NAME}_final_optimizer.pth" "$BACKBONE_OPT_MIN"
    pull "$REMOTE_DIR/runs/${NAME}_FINAL.pth"           "$LOCAL_DIR/runs/${NAME}_FINAL.pth"           "$BACKBONE_MIN"
    # Per-arm launcher log.
    pull "$REMOTE_DIR/results/dl_${arm}.log" "$LOCAL_DIR/results/dl_${arm}.log" "$TEXT_MIN"
  done
  # Orchestrator + monitor logs, one per wave.
  for w in 1 2 3; do
    pull "$REMOTE_DIR/results/orchestrate_wave${w}.log"        "$LOCAL_DIR/results/orchestrate_wave${w}.log"        "$TEXT_MIN"
    pull "$REMOTE_DIR/results/orchestrate_wave${w}_state.json" "$LOCAL_DIR/results/orchestrate_wave${w}_state.json" "$TEXT_MIN"
  done
  pull "$REMOTE_DIR/results/monitor.log" "$LOCAL_DIR/results/monitor.log" "$TEXT_MIN"

  # Optional single-arm early exit.
  if [ -n "$FINAL_SENTINEL" ] && [ -f "$FINAL_SENTINEL" ]; then
    log "FINAL sentinel $FINAL_SENTINEL present — sync loop exiting"
    exit 0
  fi
  log "--- sleeping ${INTERVAL}s ---"
  sleep "$INTERVAL"
done
