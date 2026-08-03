#!/bin/bash
# #379 orchestrator — sequences the 6 backbone-only arms across the two
# 4090 GPUs on elisa.
#
# Pipeline shape (backbones only — no q-head, no GIFT-Eval):
#
#   Phase A | arm 1 on GPU 0 || arm 3 on GPU 1
#   Phase B | arm 4 on GPU 0 || arm 5 on GPU 1
#   Phase C | arm 6_v2 on GPU 0 || bimoco on GPU 1
#
# Each backbone at d_model=64, bs=64, 200k steps takes ~15-20 h on a 4090.
# Total wall clock ≈ 3 phases × ~18 h ≈ 2.5 days.
#
# `run_arm.sh` is idempotent: a completed arm short-circuits on FINAL,
# so re-running the orchestrator after a crash resumes from whatever's
# on disk. Per-arm log stream is `results/dl_<arm>.log`; this
# orchestrator's own progress lands in `results/orchestrate.log`.
#
# Usage:
#   WT=$HOME/workspaces/contrastive-forecasting \
#     nohup setsid bash orchestrate.sh > orchestrate.log 2>&1 &
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
LOG="$RES/orchestrate.log"
STATE="$RES/orchestrate_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-379] $*" | tee -a "$LOG"; }

# Backbone base names — must match run_arm.sh's case block.
BB_arm1="bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm3="bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm4="bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm5="bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm6_v2="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_bimoco="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

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

log "orchestrator start — WT=$WT"

# ---- Phase A: arm 1 + arm 3 ------------------------------------------------
log "PHASE A: arm 1 on GPU 0, arm 3 on GPU 1"
launch_arm arm1 0 & pid_a=$!
launch_arm arm3 1 & pid_b=$!
wait_pair $pid_a $pid_b A || log "phase A had failing arm(s) — continuing to phase B (arms are independent)"

# ---- Phase B: arm 4 + arm 5 ------------------------------------------------
log "PHASE B: arm 4 on GPU 0, arm 5 on GPU 1"
launch_arm arm4 0 & pid_a=$!
launch_arm arm5 1 & pid_b=$!
wait_pair $pid_a $pid_b B || log "phase B had failing arm(s) — continuing to phase C"

# ---- Phase C: arm 6 v2 + bimoco --------------------------------------------
log "PHASE C: arm 6_v2 on GPU 0, bimoco on GPU 1"
launch_arm arm6_v2 0 & pid_a=$!
launch_arm bimoco  1 & pid_b=$!
wait_pair $pid_a $pid_b C || log "phase C had failing arm(s)"

# Summary — count of arms whose FINAL.pth landed.
count_final=0
for var in BB_arm1 BB_arm3 BB_arm4 BB_arm5 BB_arm6_v2 BB_bimoco; do
  bb="${!var}"
  [ -f "$OUT/runs/${bb}_FINAL.pth" ] && count_final=$((count_final + 1))
done
log "orchestrator done — backbones FINAL: $count_final / 6"

cat > "$STATE" <<EOF
{
  "state": "done",
  "backbones_final": $count_final,
  "expected_backbones": 6
}
EOF
log "state written to $STATE"
