#!/bin/bash
# #379 tau_rep=1.0 rerun orchestrator — sequences the 5 backbone-only
# `_tr1` arms across the two 4090 GPUs on elisa. Intended to be launched
# BY THE RESEARCHER/ORCHESTRATOR AFTER the base 6-arm sweep
# (`orchestrate.sh`) finishes; no auto-trigger.
#
# Pipeline shape (backbones only — no q-head, no GIFT-Eval):
#
#   Phase D | arm1_tr1     on GPU 0 || arm3_tr1     on GPU 1
#   Phase E | arm5_tr1     on GPU 0 || arm6_v2_tr1  on GPU 1
#   Phase F | bimoco_tr1   on GPU 0
#
# Each backbone at d_model=64, bs=64, 200k steps takes ~15-20 h on a 4090.
# Total wall clock ≈ 2 pairs + 1 solo ≈ ~3 × 18 h ≈ 2 days.
#
# `run_arm.sh` is idempotent: a completed arm short-circuits on FINAL,
# so re-running the orchestrator after a crash resumes from whatever's
# on disk. Per-arm log stream is `results/dl_<arm>.log`; this
# orchestrator's own progress lands in `results/orchestrate_tau_rep.log`.
#
# Usage (typically triggered manually AFTER `orchestrate.sh` completes):
#   WT=$HOME/workspaces/contrastive-forecasting \
#     nohup setsid bash orchestrate_tau_rep.sh > orchestrate_tau_rep.log 2>&1 &
set -uo pipefail

WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
case "$WT" in
  /tmp/*|/tmp)
    echo "ABORT: WT=$WT is under /tmp — refusing." >&2
    exit 2
    ;;
esac

OUT="$WT/experiments/2026-07-21_split_pred_rep_small"
RES="$OUT/results"; mkdir -p "$RES"
SCRIPTS="$OUT/scripts"
LOG="$RES/orchestrate_tau_rep.log"
STATE="$RES/orchestrate_tau_rep_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-379-tr1] $*" | tee -a "$LOG"; }

# Backbone base names — must match run_arm.sh's case block.
BB_arm1_tr1="bb_small_arm1_tr1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm3_tr1="bb_small_arm3_tr1_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm5_tr1="bb_small_arm5_tr1_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm6_v2_tr1="bb_small_arm6_v2_tr1_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_bimoco_tr1="bb_small_bimoco_tr1_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

launch_arm(){ # arm bb_gpu
  local arm="$1" bb_gpu="$2"
  log "arm $arm start: BB_GPU=$bb_gpu"
  WT="$WT" BB_GPU="$bb_gpu" bash "$SCRIPTS/run_arm.sh" "$arm" >>"$LOG" 2>&1
  local rc=$?
  log "arm $arm done rc=$rc"
  return $rc
}

wait_pair(){ # pid_a pid_b tag
  local pa="$1" pb="$2" tag="$3"
  wait "$pa"; local ra=$?
  wait "$pb"; local rb=$?
  log "phase $tag joined: pids=$pa/$pb rc=$ra/$rb"
  return $(( ra != 0 || rb != 0 ))
}

log "orchestrator start — WT=$WT (tau_rep=1.0 reruns)"

# ---- Phase D: arm1_tr1 + arm3_tr1 ------------------------------------------
log "PHASE D: arm1_tr1 on GPU 0, arm3_tr1 on GPU 1"
launch_arm arm1_tr1 0 & pid_a=$!
launch_arm arm3_tr1 1 & pid_b=$!
wait_pair $pid_a $pid_b D || log "phase D had failing arm(s) — continuing to phase E (arms are independent)"

# ---- Phase E: arm5_tr1 + arm6_v2_tr1 ---------------------------------------
log "PHASE E: arm5_tr1 on GPU 0, arm6_v2_tr1 on GPU 1"
launch_arm arm5_tr1 0 & pid_a=$!
launch_arm arm6_v2_tr1 1 & pid_b=$!
wait_pair $pid_a $pid_b E || log "phase E had failing arm(s) — continuing to phase F"

# ---- Phase F: bimoco_tr1 (solo — odd count of 5) ---------------------------
# Two-GPU pipeline leaves one arm alone in the last phase; put it on GPU 0
# (arbitrary, symmetric with `orchestrate.sh`'s Phase A layout).
log "PHASE F: bimoco_tr1 on GPU 0"
launch_arm bimoco_tr1 0
rc_f=$?
log "phase F done rc=$rc_f"

# Summary — count of arms whose FINAL.pth landed.
count_final=0
for var in BB_arm1_tr1 BB_arm3_tr1 BB_arm5_tr1 BB_arm6_v2_tr1 BB_bimoco_tr1; do
  bb="${!var}"
  [ -f "$OUT/runs/${bb}_FINAL.pth" ] && count_final=$((count_final + 1))
done
log "orchestrator done — tau_rep backbones FINAL: $count_final / 5"

cat > "$STATE" <<EOF
{
  "state": "done",
  "backbones_final": $count_final,
  "expected_backbones": 5
}
EOF
log "state written to $STATE"
