#!/bin/bash
# #366 — sweep driver: train each cross arm's backbone and then run its
# downstream cells. Arms run sequentially on the same GPU pair; per-arm
# downstream uses both GPUs in parallel (2L on GPU_2L, 6L on GPU_6L).
#
# The arm identities (λ pair × τ) are NOT baked in. They are read from a
# winners manifest written at launch time — see `winners.sh.example` for
# the format and re-verify procedure. The issue (#366) requires the λ
# pairs and τ to be picked at launch from #363 / #357's final state, and
# launch is gated on #363 closing.
#
#   launch_arms.sh                     run both arms
#   ONLY="lA_..." launch_arms.sh       pick a subset by suffix
#   WINNERS_FILE=... launch_arms.sh    override the manifest path
set -uo pipefail
: "${WT:?WT (worktree root containing experiments/<this-dir>/scripts/...) must be set; e.g. WT=/home/jupyter/workspaces/contrastive-forecasting}"
: "${OUT:?OUT (per-experiment output dir for runs/ and results/) must be set; e.g. OUT=\$WT/experiments/2026-06-28_sigreg_lambda_tau_cross}"
GPU="${GPU:-0}"
export WT OUT
[ -d "$WT" ] || { echo "[cross] ABORT: WT does not exist: $WT" >&2; exit 2; }
EXP_SCRIPTS="$WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts"
BB_SCRIPT="$EXP_SCRIPTS/train_backbone_sigreg.sh"
DL_SCRIPT="$EXP_SCRIPTS/launch_downstream.sh"
WINNERS_EXAMPLE="$EXP_SCRIPTS/winners.sh.example"
[ -f "$BB_SCRIPT" ] || { echo "[cross] ABORT: BB_SCRIPT not found: $BB_SCRIPT" >&2; exit 2; }
[ -f "$DL_SCRIPT" ] || { echo "[cross] ABORT: DL_SCRIPT not found: $DL_SCRIPT" >&2; exit 2; }
RES="$OUT/results"; mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [cross] $*"; }

# Launch-time gate: require a manifest of the verified λ pairs × τ. The
# manifest is created by the user after #363 closes and #357 is final
# (see winners.sh.example).
WINNERS_FILE="${WINNERS_FILE:-$OUT/winners.sh}"
if [ ! -f "$WINNERS_FILE" ]; then
  cat >&2 <<EOF
[cross] ABORT: winners manifest not found at $WINNERS_FILE.

Issue #366 requires the λ pair × τ to be re-verified at launch time
because #363 / #357 winners may shift between scaffold and launch.

To proceed:
  cp $WINNERS_EXAMPLE $WINNERS_FILE
  \$EDITOR $WINNERS_FILE       # fill in values + verification stamps
  bash $0
EOF
  exit 2
fi
# shellcheck disable=SC1090
. "$WINNERS_FILE"
for v in ARM_A_LAMBDA_E ARM_A_LAMBDA_H ARM_B_LAMBDA_E ARM_B_LAMBDA_H BEST_TAU \
         WINNERS_VERIFIED_BY WINNERS_VERIFIED_AT; do
  if [ -z "${!v:-}" ]; then
    echo "[cross] ABORT: $v is unset/empty in $WINNERS_FILE — re-verify and stamp." >&2
    exit 2
  fi
done
log "winners verified by ${WINNERS_VERIFIED_BY} on ${WINNERS_VERIFIED_AT}"
log "Arm A (best-at-best): λ_e=${ARM_A_LAMBDA_E} λ_h=${ARM_A_LAMBDA_H} τ=${BEST_TAU}"
log "Arm B (best-at-last): λ_e=${ARM_B_LAMBDA_E} λ_h=${ARM_B_LAMBDA_H} τ=${BEST_TAU}"

# Derive a per-arm suffix from (prefix, λ_e, λ_h, τ). Encodes λ × 10 and
# τ × 100, matching the #363 / #357 naming convention — a stale manifest
# changes the suffix, so wrong values do not silently overwrite a prior
# run's files. Public so the consistency test can shell out to it.
#
# `%.0f` rounds; `%d` would truncate toward zero. FP error pushes e.g.
# 0.58 × 100 to 57.9999…, which `%d` mis-encodes as `tau057`, defeating
# the stale-value-changes-suffix guarantee. Rounding round-trips.
suffix_for(){ # prefix lambda_e lambda_h tau
  awk -v p="$1" -v le="$2" -v lh="$3" -v t="$4" \
    'BEGIN { printf "%s_emb%.0f_enc%.0f_tau%03.0f\n", p, le*10, lh*10, t*100 }'
}

SUFFIX_A=$(suffix_for lA "$ARM_A_LAMBDA_E" "$ARM_A_LAMBDA_H" "$BEST_TAU")
SUFFIX_B=$(suffix_for lB "$ARM_B_LAMBDA_E" "$ARM_B_LAMBDA_H" "$BEST_TAU")

ARMS=(
  "$ARM_A_LAMBDA_E $ARM_A_LAMBDA_H $BEST_TAU $SUFFIX_A"
  "$ARM_B_LAMBDA_E $ARM_B_LAMBDA_H $BEST_TAU $SUFFIX_B"
)

run_arm(){ # lambda_e lambda_h tau suffix
  local le="$1" lh="$2" tau="$3" suffix="$4"
  log "ARM ${suffix} λ_e=${le} λ_h=${lh} τ=${tau} — backbone start"
  bash "$BB_SCRIPT" "$GPU" "$le" "$lh" "$tau" "$suffix" >>"$RES/sweep_bb_${suffix}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "ARM ${suffix} backbone FAILED rc=$rc — skipping downstream"
    return $rc
  fi
  log "ARM ${suffix} — downstream start"
  bash "$DL_SCRIPT" "$suffix" >>"$RES/sweep_dl_${suffix}.log" 2>&1
  rc=$?
  log "ARM ${suffix} downstream rc=$rc"
  return $rc
}

failed=0
for entry in "${ARMS[@]}"; do
  read -r le lh tau suffix <<< "$entry"
  if [ -n "${ONLY:-}" ] && ! echo " $ONLY " | grep -q " $suffix "; then
    log "ARM ${suffix} skipped by ONLY filter"
    continue
  fi
  run_arm "$le" "$lh" "$tau" "$suffix" || failed=$((failed+1))
done
log "cross done; failed arms: $failed"
exit "$failed"
