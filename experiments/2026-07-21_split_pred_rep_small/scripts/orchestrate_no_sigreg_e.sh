#!/bin/bash
# #379 no-sigreg-embedding (`_nse`) rerun orchestrator — STAGED WAVES.
#
# The 6 `_nse` arms are trained in three waves (identical schedule to
# `orchestrate_tau_rep.sh`). Every wave brings all 6 arms to the next
# milestone before any arm advances further; this feeds the Wave-D-first
# barrier the researcher asked for on the 23-arm sweep.
#
# Phase letters G/H/I are chosen to avoid collision with the base 6-arm
# orchestrator's A/B/C and the tau_rep orchestrator's D/E/F in a shared
# log — the ncpc orchestrator continues with J/K/L, and the fresh-base
# orchestrator with M/N/O.
#
#   PHASE G (wave 1) → step  40 000 (save-every 10 000, extras {2500, 40000})
#   PHASE H (wave 2) → step 100 000 (save-every 25 000, extra  {100000})
#   PHASE I (wave 3) → step 200 000 (save-every 25 000, no extras)
#
# Within each PHASE the 6 arms are pipelined across the 2× 4090 as three
# sub-phases (letters follow the parent PHASE — G1/G2/G3 belong to
# PHASE G, etc.):
#
#     sub-phase X1 | arm1_nse    on GPU 0 || arm3_nse   on GPU 1
#     sub-phase X2 | arm4_nse    on GPU 0 || arm5_nse   on GPU 1
#     sub-phase X3 | arm6_v2_nse on GPU 0 || bimoco_nse on GPU 1
#
# All 6 arms in a phase finish before the next phase starts (barrier per
# phase — the wall clock of a phase = its slowest arm, not sum-of-arms).
#
# `run_arm.sh` accepts TARGET_STEPS / FINAL_STEPS: only the final wave
# writes `_FINAL.pth`; intermediate waves leave the `_<N>k.pth` on disk
# so the next wave's launcher resumes from it via `--resume`.
#
# `run_arm.sh` is idempotent per wave: a wave whose target checkpoint
# already exists short-circuits (WAVE SKIP), so re-running the
# orchestrator after a crash resumes from whatever's on disk.
#
# Per-arm training log stream: `results/dl_<arm>.log`.
# Orchestrator progress:       `results/orchestrate_no_sigreg_e.log`.
#
# Usage:
#   WT=$HOME/workspaces/contrastive-forecasting \
#     nohup setsid bash orchestrate_no_sigreg_e.sh > orchestrate_no_sigreg_e.log 2>&1 &
#
# Optional env:
#   MAX_WAVE=G          # break after PHASE G (Wave-D barrier: only reach 40k)
#                       #   unset → run all waves (G → H → I)
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
LOG="$RES/orchestrate_no_sigreg_e.log"
STATE="$RES/orchestrate_no_sigreg_e_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-379-nse] $*" | tee -a "$LOG"; }

# Backbone base names — must match run_arm.sh's case block.
BB_arm1_nse="bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm3_nse="bb_small_arm3_nse_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm4_nse="bb_small_arm4_nse_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm5_nse="bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_arm6_v2_nse="bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_bimoco_nse="bb_small_bimoco_nse_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

FINAL_STEPS=200000

# Optional early-exit letter (Wave-D barrier). Unset → run every wave.
MAX_WAVE="${MAX_WAVE:-}"

# Wave schedule — one entry per wave / phase. Fields:
#   wave_index | phase_letter | TARGET_STEPS | SAVE_EVERY | EXTRA_SAVES
# Phase letters G/H/I so a shared orchestrator log is unambiguous.
#   phase G (wave 1): 40k,  save-every 10k, extras 2500 + 40000  → _40k.pth
#   phase H (wave 2): 100k, save-every 25k, extra  100000        → _100k.pth
#   phase I (wave 3): 200k, save-every 25k, no extras            → _200k.pth
WAVE_SCHEDULE=(
  "1|G|40000|10000|2500,40000"
  "2|H|100000|25000|100000"
  "3|I|200000|25000|"
)

launch_arm(){ # arm bb_gpu phase_letter target save_every extras
  local arm="$1" bb_gpu="$2" letter="$3" target="$4" se="$5" ex="$6"
  log "PHASE $letter · arm $arm start: BB_GPU=$bb_gpu target=$target save_every=$se extras=[$ex]"
  WT="$WT" BB_GPU="$bb_gpu" \
    TARGET_STEPS="$target" FINAL_STEPS="$FINAL_STEPS" \
    SAVE_EVERY="$se" EXTRA_SAVES="$ex" \
    bash "$SCRIPTS/run_arm.sh" "$arm" >>"$LOG" 2>&1
  local rc=$?
  log "PHASE $letter · arm $arm done rc=$rc"
  return $rc
}

wait_pair(){ # pid_a pid_b tag
  local pa="$1" pb="$2" tag="$3"
  wait "$pa"; local ra=$?
  wait "$pb"; local rb=$?
  log "$tag joined: pids=$pa/$pb rc=$ra/$rb"
  return $(( ra != 0 || rb != 0 ))
}

# Count how many nse arms have a periodic checkpoint at ≥ target_k.
count_arms_at_step(){ # step_k
  local step_k="$1" n=0
  for var in BB_arm1_nse BB_arm3_nse BB_arm4_nse BB_arm5_nse BB_arm6_v2_nse BB_bimoco_nse; do
    local bb="${!var}"
    local best=-1
    for f in "$OUT/runs/${bb}"_*k.pth; do
      [ -e "$f" ] || continue
      case "$f" in *_optimizer.pth) continue;; esac
      local k
      k=$(basename "$f" | sed -E 's/.*_([0-9]+)k\.pth$/\1/')
      case "$k" in ''|*[!0-9]*) continue;; esac
      (( k > best )) && best=$k
    done
    (( best >= step_k )) && n=$(( n + 1 ))
  done
  echo "$n"
}

run_wave(){ # wave letter target save_every extras
  local wave="$1" letter="$2" target="$3" se="$4" ex="$5"
  log "PHASE $letter START — wave $wave · target=$target save_every=$se extras=[$ex]"

  log "PHASE $letter · sub-phase ${letter}1: arm1_nse (GPU 0) + arm3_nse (GPU 1)"
  launch_arm arm1_nse    0 "$letter" "$target" "$se" "$ex" & pid_a=$!
  launch_arm arm3_nse    1 "$letter" "$target" "$se" "$ex" & pid_b=$!
  wait_pair $pid_a $pid_b "PHASE $letter · sub-phase ${letter}1" \
    || log "PHASE $letter · sub-phase ${letter}1 had failing arm(s) — continuing to ${letter}2"

  log "PHASE $letter · sub-phase ${letter}2: arm4_nse (GPU 0) + arm5_nse (GPU 1)"
  launch_arm arm4_nse    0 "$letter" "$target" "$se" "$ex" & pid_a=$!
  launch_arm arm5_nse    1 "$letter" "$target" "$se" "$ex" & pid_b=$!
  wait_pair $pid_a $pid_b "PHASE $letter · sub-phase ${letter}2" \
    || log "PHASE $letter · sub-phase ${letter}2 had failing arm(s) — continuing to ${letter}3"

  log "PHASE $letter · sub-phase ${letter}3: arm6_v2_nse (GPU 0) + bimoco_nse (GPU 1)"
  launch_arm arm6_v2_nse 0 "$letter" "$target" "$se" "$ex" & pid_a=$!
  launch_arm bimoco_nse  1 "$letter" "$target" "$se" "$ex" & pid_b=$!
  wait_pair $pid_a $pid_b "PHASE $letter · sub-phase ${letter}3" \
    || log "PHASE $letter · sub-phase ${letter}3 had failing arm(s)"

  local n_reached
  n_reached=$(count_arms_at_step "$(( target / 1000 ))")
  log "PHASE $letter DONE — wave $wave · arms at ≥ ${target} steps: $n_reached / 6"
}

log "orchestrator start — WT=$WT (no_sigreg_e reruns · staged waves · MAX_WAVE='${MAX_WAVE}')"
log "wave schedule: $(printf '%s ' "${WAVE_SCHEDULE[@]}")"

for entry in "${WAVE_SCHEDULE[@]}"; do
  IFS='|' read -r WAVE LETTER TARGET SAVE_EVERY EXTRAS <<< "$entry"
  run_wave "$WAVE" "$LETTER" "$TARGET" "$SAVE_EVERY" "$EXTRAS"
  if [ -n "$MAX_WAVE" ] && [ "$LETTER" = "$MAX_WAVE" ]; then
    log "MAX_WAVE=$MAX_WAVE reached — stopping after PHASE $LETTER"
    break
  fi
done

# Summary — count of arms whose FINAL.pth landed (only written after
# wave 3 reaches step FINAL_STEPS).
count_final=0
for var in BB_arm1_nse BB_arm3_nse BB_arm4_nse BB_arm5_nse BB_arm6_v2_nse BB_bimoco_nse; do
  bb="${!var}"
  [ -f "$OUT/runs/${bb}_FINAL.pth" ] && count_final=$((count_final + 1))
done
log "orchestrator done — nse backbones FINAL: $count_final / 6"

cat > "$STATE" <<EOF
{
  "state": "done",
  "backbones_final": $count_final,
  "expected_backbones": 6,
  "waves": ["40000", "100000", "200000"],
  "max_wave": "${MAX_WAVE}"
}
EOF
log "state written to $STATE"
