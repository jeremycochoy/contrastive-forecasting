#!/bin/bash
# #379 persistent sync loop — 15-min ticks pulling backbone checkpoints,
# losses.csv, and training logs from the elisa training host into a
# permanent local checkout. Written to satisfy CLAUDE.md § Remote
# Machine Monitoring: EVERY remote run has a sync loop for its full
# duration, atomic writes only, per-class size thresholds, ls-verify
# not log-verify.
#
# This experiment trains 6 backbone-only arms sequentially on elisa
# (see orchestrate.sh). Each arm writes to $REMOTE_DIR/runs/${NAME}_*
# and $REMOTE_DIR/results/*; the loop iterates over the 6 arm-name
# prefixes and pulls whatever is there. It never blocks on a missing
# file (safe_pull.sh returns non-zero and leaves the local copy
# untouched).
#
# Usage on the machine that OWNS the local persistent checkout:
#   REMOTE_HOST=elisa REMOTE_DIR=~/workspaces/contrastive-forecasting/experiments/2026-07-21_split_pred_rep_small \
#   LOCAL_DIR=/absolute/path/to/experiments/2026-07-21_split_pred_rep_small \
#     nohup setsid bash sync_loop.sh > sync_loop.log 2>&1 &
set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:?REMOTE_HOST must be set (e.g. elisa or ssh5.vast.ai)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:?REMOTE_DIR must be set (path on REMOTE_HOST)}"
LOCAL_DIR="${LOCAL_DIR:?LOCAL_DIR must be set (absolute path in the persistent checkout)}"
INTERVAL="${INTERVAL:-900}"        # 15-min ticks (CLAUDE.md)
FINAL_SENTINEL="${FINAL_SENTINEL:-}"  # optional single-arm early-exit

case "$LOCAL_DIR" in
  /tmp/*|/tmp)
    echo "ABORT: LOCAL_DIR=$LOCAL_DIR is under /tmp — refusing to sync into an ephemeral path." >&2
    exit 2
    ;;
esac

# safe_pull.sh: atomic scp with .tmp → mv, .prev rotation, per-file size
# floor. Lives in the periodic-synth-mix scripts dir (added PR #45).
SAFE_PULL="$(dirname "$LOCAL_DIR")/../2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"
[ -f "$SAFE_PULL" ] || {
  echo "ABORT: safe_pull.sh not at $SAFE_PULL" >&2
  exit 2
}

ARMS=(arm1 arm3 arm4 arm5 arm6_v2 bimoco)
# Base run names must match the case block in run_arm.sh — kept in sync
# manually because both files are short and edits to either are rare.
NAME_arm1="bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
NAME_arm3="bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
NAME_arm4="bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
NAME_arm5="bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
NAME_arm6_v2="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
NAME_bimoco="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

# Per-class minimum-byte floors (never one blanket number — CLAUDE.md).
# Small backbone (d_model=64, 3+3 layers, ~1-2M params) → backbone
# checkpoint ~5 MB and optimizer state ~10 MB. CSVs a few KB, log a few KB.
BACKBONE_MIN=3000000         # ~5 MB actual, floor 3 MB
BACKBONE_OPT_MIN=6000000     # ~10 MB actual, floor 6 MB
TEXT_MIN=100                 # CSVs / logs: at least a header

# Backbone snapshot step_k values on disk: extra-save at 2500 (`_2k.pth`)
# plus save-every=25000 cadence out to 200k.
BACKBONE_STEPS_K="2 25 50 75 100 125 150 175 200"

pull(){ # remote_path local_path min_bytes
  bash "$SAFE_PULL" "$REMOTE_HOST" "$REMOTE_PORT" "$1" "$2" "$3" \
    || echo "  (skip: $1 not present or below floor)"
}

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [sync-379] $*"; }

log "start REMOTE=$REMOTE_HOST:$REMOTE_DIR LOCAL=$LOCAL_DIR interval=${INTERVAL}s"
mkdir -p "$LOCAL_DIR/runs" "$LOCAL_DIR/results"

# Loop forever unless FINAL_SENTINEL is set and appears locally.
while true; do
  log "tick"
  for arm in "${ARMS[@]}"; do
    varname="NAME_$arm"
    NAME="${!varname}"
    # Best-checkpoint pairs (train.py writes on each new best-loss / best-gap).
    for f in best_gap.pth best_loss.pth; do
      pull "$REMOTE_DIR/runs/${NAME}_$f" \
           "$LOCAL_DIR/runs/${NAME}_$f" "$BACKBONE_MIN"
      pull "$REMOTE_DIR/runs/${NAME}_${f%.pth}_optimizer.pth" \
           "$LOCAL_DIR/runs/${NAME}_${f%.pth}_optimizer.pth" "$BACKBONE_OPT_MIN"
    done
    # Losses / attention-amplitude CSVs, training log.
    pull "$REMOTE_DIR/runs/${NAME}_losses.csv"          "$LOCAL_DIR/runs/${NAME}_losses.csv"          "$TEXT_MIN"
    pull "$REMOTE_DIR/runs/${NAME}_attn_amplitude.csv"  "$LOCAL_DIR/runs/${NAME}_attn_amplitude.csv"  "$TEXT_MIN"
    pull "$REMOTE_DIR/results/run_${NAME}.log"          "$LOCAL_DIR/results/run_${NAME}.log"          "$TEXT_MIN"
    # Periodic step snapshots (backbone + optimizer). safe_pull.sh silently
    # skips missing files, so listing every possible step is fine.
    for sk in $BACKBONE_STEPS_K; do
      pull "$REMOTE_DIR/runs/${NAME}_${sk}k.pth"           "$LOCAL_DIR/runs/${NAME}_${sk}k.pth"           "$BACKBONE_MIN"
      pull "$REMOTE_DIR/runs/${NAME}_${sk}k_optimizer.pth" "$LOCAL_DIR/runs/${NAME}_${sk}k_optimizer.pth" "$BACKBONE_OPT_MIN"
    done
    # Restart-suffixed variants (safe_run_name adds _r2, _r3, …). Only
    # pull the two most common (rare > _r3 in practice).
    for rk in _r2 _r3; do
      pull "$REMOTE_DIR/runs/${NAME}${rk}_final.pth"           "$LOCAL_DIR/runs/${NAME}${rk}_final.pth"           "$BACKBONE_MIN"
      pull "$REMOTE_DIR/runs/${NAME}${rk}_final_optimizer.pth" "$LOCAL_DIR/runs/${NAME}${rk}_final_optimizer.pth" "$BACKBONE_OPT_MIN"
      pull "$REMOTE_DIR/runs/${NAME}${rk}_losses.csv"          "$LOCAL_DIR/runs/${NAME}${rk}_losses.csv"          "$TEXT_MIN"
    done
    # Final backbone snapshot + launcher-copied FINAL sentinel.
    pull "$REMOTE_DIR/runs/${NAME}_final.pth"           "$LOCAL_DIR/runs/${NAME}_final.pth"           "$BACKBONE_MIN"
    pull "$REMOTE_DIR/runs/${NAME}_final_optimizer.pth" "$LOCAL_DIR/runs/${NAME}_final_optimizer.pth" "$BACKBONE_OPT_MIN"
    pull "$REMOTE_DIR/runs/${NAME}_FINAL.pth"           "$LOCAL_DIR/runs/${NAME}_FINAL.pth"           "$BACKBONE_MIN"
    # Per-arm launcher log.
    pull "$REMOTE_DIR/results/dl_${arm}.log" "$LOCAL_DIR/results/dl_${arm}.log" "$TEXT_MIN"
  done
  # Orchestrator log lives at the experiment root.
  pull "$REMOTE_DIR/results/orchestrate.log" "$LOCAL_DIR/results/orchestrate.log" "$TEXT_MIN"

  # Optional single-arm early exit (used by smoke.sh).
  if [ -n "$FINAL_SENTINEL" ] && [ -f "$FINAL_SENTINEL" ]; then
    log "FINAL sentinel $FINAL_SENTINEL present — sync loop exiting"
    exit 0
  fi
  log "--- sleeping ${INTERVAL}s ---"
  sleep "$INTERVAL"
done
