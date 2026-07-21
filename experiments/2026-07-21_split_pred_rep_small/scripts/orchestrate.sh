#!/bin/bash
# #379 orchestrator — sequences the 6 arms across the two 4090 GPUs on elisa.
#
# Pipeline shape (matches the 4-5-day estimate in the issue):
#
#   Phase A backbones      | arm 1 on GPU 0 || arm 3 on GPU 1     (~35 h)
#   Phase A downstream     | arm 1 dl 2L on GPU 0 || arm 1 dl 6L on GPU 1
#                            arm 3 dl 2L on GPU 0 || arm 3 dl 6L on GPU 1
#   Phase B backbones      | arm 4 on GPU 0 || arm 5 on GPU 1     (~35 h)
#   Phase B downstream     | arm 4 dl 2L on GPU 0 || arm 4 dl 6L on GPU 1
#                            arm 5 dl 2L on GPU 0 || arm 5 dl 6L on GPU 1
#   Phase C backbones      | arm 6_v2 on GPU 0 || bimoco on GPU 1 (~35 h)
#   Phase C downstream     | arm 6_v2 dl on 0/1  || bimoco dl on 0/1
#
# So the two GPUs stay busy for both training and downstream. Total wall
# clock ≈ 3 × 35 h backbone + 3 × 10 h downstream ≈ 4.5 days.
#
# `run_arm.sh` is idempotent: a completed arm short-circuits (FINAL / summary.txt
# skip guards), so re-running the orchestrator after a crash resumes from
# whatever's on disk. Every arm's own log stream is `results/dl_<arm>.log`;
# this orchestrator's own progress lands in `results/orchestrate.log`.
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
BB_arm1="bb_small_arm1_split_pred_rep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm3="bb_small_arm3_split_pred_rep_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm4="bb_small_arm4_xshh_allt_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm5="bb_small_arm5_lalign_lrep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm6_v2="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_bimoco="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090"

# Launch one arm end-to-end with the given GPU assignment.
launch_arm(){ # arm bb_gpu gpu_2l gpu_6l
  local arm="$1" bb_gpu="$2" g2l="$3" g6l="$4"
  log "arm $arm start: BB_GPU=$bb_gpu GPU_2L=$g2l GPU_6L=$g6l"
  WT="$WT" BB_GPU="$bb_gpu" GPU_2L="$g2l" GPU_6L="$g6l" \
    bash "$SCRIPTS/run_arm.sh" "$arm" >>"$LOG" 2>&1
  local rc=$?
  log "arm $arm done rc=$rc"
  return $rc
}

# Wait for both PIDs and return non-zero iff either failed.
wait_pair(){ # pid_a pid_b tag
  local pa="$1" pb="$2" tag="$3"
  wait "$pa"; local ra=$?
  wait "$pb"; local rb=$?
  log "phase $tag joined: pids=$pa/$pb rc=$ra/$rb"
  return $(( ra != 0 || rb != 0 ))
}

# Backbone-only wrapper — sets HEAD_STEPS=0 would skip head cells, but run_arm.sh
# doesn't have that toggle; instead we let the arm progress through its
# backbone THEN downstream. The two arms in a phase run concurrently and
# each drives its 2L+6L downstream in parallel on the same two GPUs. To
# avoid a 4-cell contention within a phase we serialize downstream by
# assigning both arms the SAME GPUs and letting their outputs interleave.
#
# Simpler: launch (backbone-only) first, join, then launch (downstream-only).
# But run_arm.sh does both in one shot. So we accept that within a phase,
# arm A's downstream may overlap with arm B's backbone tail — the downstream
# cells are 5 x head-train + 5 x eval each, several hours; on the two 4090s
# this remains within the total 4-5-day budget.

log "orchestrator start — WT=$WT"

# ---- Phase A: arm 1 + arm 3 ------------------------------------------------
log "PHASE A: arm 1 on GPU 0 (dl 6L on GPU 0, dl 2L on GPU 1), arm 3 on GPU 1 (dl 6L on GPU 1, dl 2L on GPU 0)"
launch_arm arm1 0 1 0 &
pid_a=$!
launch_arm arm3 1 0 1 &
pid_b=$!
wait_pair $pid_a $pid_b A || log "phase A had failing arm(s) — continuing to phase B (arms are independent)"

# ---- Phase B: arm 4 + arm 5 ------------------------------------------------
log "PHASE B: arm 4 on GPU 0, arm 5 on GPU 1"
launch_arm arm4 0 1 0 &
pid_a=$!
launch_arm arm5 1 0 1 &
pid_b=$!
wait_pair $pid_a $pid_b B || log "phase B had failing arm(s) — continuing to phase C"

# ---- Phase C: arm 6 v2 + bimoco --------------------------------------------
log "PHASE C: arm 6_v2 on GPU 0, bimoco on GPU 1"
launch_arm arm6_v2 0 1 0 &
pid_a=$!
launch_arm bimoco  1 0 1 &
pid_b=$!
wait_pair $pid_a $pid_b C || log "phase C had failing arm(s)"

# Summary — count of arms whose FINAL.pth landed and 20-cell counts.
count_final=0; total_summary=0
for var in BB_arm1 BB_arm3 BB_arm4 BB_arm5 BB_arm6_v2 BB_bimoco; do
  bb="${!var}"
  [ -f "$OUT/runs/${bb}_FINAL.pth" ] && count_final=$((count_final + 1))
done
total_summary=$(ls "$OUT/results"/gift_eval_full_*/summary.txt 2>/dev/null | wc -l)
log "orchestrator done — backbones FINAL: $count_final / 6, downstream cells with summary.txt: $total_summary / 60"

cat > "$STATE" <<EOF
{
  "state": "done",
  "backbones_final": $count_final,
  "downstream_cells": $total_summary,
  "expected_cells": 60
}
EOF
log "state written to $STATE"
