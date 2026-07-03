#!/bin/bash
# #369 — orchestrator: read the winners manifest (λ pair × τ + parent's
# best-loss step), train the retrained B=1024 backbone, then run
# downstream at BOTH loci for 2L and 6L heads in parallel on GPUs 0/1.
#
# Single-arm (unlike #366 which had A/B and C–H): the arm identity is
# 100% picked from the manifest.
#
#   launch_experiment.sh           run the arm named by the manifest
#   BB_ONLY=1 launch_experiment.sh run just the backbone
#   DL_ONLY=1 launch_experiment.sh run just the downstream cells
#   WINNERS_FILE=... launch_experiment.sh   override the manifest path
set -uo pipefail
: "${WT:?WT (worktree root containing experiments/<this-dir>/scripts/...) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set; e.g. OUT=\$WT/experiments/2026-07-03_b1024_traj_ckpts}"
GPU_BB="${GPU_BB:-0}"
GPU_2L="${GPU_2L:-0}"
GPU_6L="${GPU_6L:-1}"
export WT OUT
[ -d "$WT" ] || { echo "[b1024] ABORT: WT does not exist: $WT" >&2; exit 2; }
EXP_SCRIPTS="$WT/experiments/2026-07-03_b1024_traj_ckpts/scripts"
BB_SCRIPT="$EXP_SCRIPTS/train_backbone_b1024.sh"
DL_SCRIPT="$EXP_SCRIPTS/downstream_b1024.sh"
WINNERS_EXAMPLE="$EXP_SCRIPTS/winners.sh.example"
[ -f "$BB_SCRIPT" ] || { echo "[b1024] ABORT: BB_SCRIPT not found: $BB_SCRIPT" >&2; exit 2; }
[ -f "$DL_SCRIPT" ] || { echo "[b1024] ABORT: DL_SCRIPT not found: $DL_SCRIPT" >&2; exit 2; }
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [b1024] $*"; }

# Launch-time gate: manifest must exist and be stamped. Reason: the
# winner is picked at launch from #366's final gm_table.csv, which is
# still in flight at scaffold time.
WINNERS_FILE="${WINNERS_FILE:-$OUT/winners.sh}"
if [ ! -f "$WINNERS_FILE" ]; then
  cat >&2 <<EOF
[b1024] ABORT: winners manifest not found at $WINNERS_FILE.

Issue #369 requires the λ pair × τ AND the parent's best-loss step to
be locked at launch time from #366's final gm_table.csv (arms A–I).

To proceed:
  cp $WINNERS_EXAMPLE $WINNERS_FILE
  \$EDITOR $WINNERS_FILE       # fill in values + verification stamps
  bash $0
EOF
  exit 2
fi
# shellcheck disable=SC1090
. "$WINNERS_FILE"
for v in LAMBDA_E LAMBDA_H TAU PARENT_BEST_LOSS_STEP \
         WINNERS_VERIFIED_BY WINNERS_VERIFIED_AT; do
  if [ -z "${!v:-}" ]; then
    echo "[b1024] ABORT: $v is unset/empty in $WINNERS_FILE — re-verify and stamp." >&2
    exit 2
  fi
done
log "winners verified by ${WINNERS_VERIFIED_BY} on ${WINNERS_VERIFIED_AT}"
log "Arm: λ_e=${LAMBDA_E} λ_h=${LAMBDA_H} τ=${TAU} B=1024 parent_best_loss_step=${PARENT_BEST_LOSS_STEP}"

# Derive suffix from the manifest, matching #366's encoding
# (`emb<10·λ_e>_enc<10·λ_h>_tau<100·τ>`) with a `_b1024` marker appended.
# The `_b1024` marker is what distinguishes this run's checkpoints from
# the parent's B=512 files if both live in the same runs/ dir.
#
# `%.0f` rounds; `%d` would truncate and mis-encode τ=0.58 → tau057.
suffix_for(){ # lambda_e lambda_h tau
  awk -v le="$1" -v lh="$2" -v t="$3" \
    'BEGIN { printf "l_emb%.0f_enc%.0f_tau%03.0f_b1024\n", le*10, lh*10, t*100 }'
}
SUFFIX=$(suffix_for "$LAMBDA_E" "$LAMBDA_H" "$TAU")
log "derived suffix: ${SUFFIX}"

STEPS="${STEPS:-12500}"
TRAJ_SAVE_EVERY="${TRAJ_SAVE_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-2500}"

# Backbone (one GPU).
if [ "${DL_ONLY:-0}" != 1 ]; then
  log "BB phase — GPU $GPU_BB"
  bash "$BB_SCRIPT" "$GPU_BB" "$LAMBDA_E" "$LAMBDA_H" "$TAU" "$SUFFIX" \
                   "$STEPS" "$SAVE_EVERY" "$TRAJ_SAVE_EVERY" \
                   >>"$RES/sweep_bb_${SUFFIX}.log" 2>&1
  rc_bb=$?
  log "BB rc=$rc_bb"
  [ $rc_bb -eq 0 ] || { log "ABORT: backbone failure"; exit 1; }
fi

# Downstream (2L + 6L in parallel on separate GPUs; each cell writes
# heads for BOTH loci).
if [ "${BB_ONLY:-0}" != 1 ]; then
  log "DL phase — 2L on g${GPU_2L}, 6L on g${GPU_6L} (parallel)"
  bash "$DL_SCRIPT" 2 "$GPU_2L" "$SUFFIX" "$PARENT_BEST_LOSS_STEP" "$STEPS" \
                   >>"$RES/sweep_dl_${SUFFIX}_2L.log" 2>&1 & pid2=$!
  bash "$DL_SCRIPT" 6 "$GPU_6L" "$SUFFIX" "$PARENT_BEST_LOSS_STEP" "$STEPS" \
                   >>"$RES/sweep_dl_${SUFFIX}_6L.log" 2>&1 & pid6=$!
  log "DL pids: 2L=$pid2  6L=$pid6"
  wait $pid2; rc2=$?
  wait $pid6; rc6=$?
  log "DL done: 2L rc=$rc2  6L rc=$rc6"
  exit $((rc2 + rc6))
fi
