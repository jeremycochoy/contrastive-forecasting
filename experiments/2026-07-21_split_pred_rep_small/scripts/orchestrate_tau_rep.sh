#!/bin/bash
# #379 tau_rep=1.0 rerun orchestrator — STAGED WAVES version.
#
# The 5 tau_rep arms are trained in three waves rather than each to 200k
# end-to-end. Every wave brings all 5 arms to the next milestone before
# any arm advances further, yielding early cross-arm comparisons of
# `1 − ff`. Each wave is a top-level PHASE (letters continue from the
# base 6-arm orchestrator's A/B/C):
#
#   PHASE D (wave 1) → step  40 000 (save-every 10 000, extras {2500, 40000})
#   PHASE E (wave 2) → step 100 000 (save-every 25 000, extra  {100000})
#   PHASE F (wave 3) → step 200 000 (save-every 25 000, no extras)
#
# Within each PHASE the 5 arms are pipelined across the 2× 4090 as three
# sub-phases (matches the base 6-arm orchestrator's shape), named after
# the parent phase letter — sub-phases D1/D2/D3 belong to PHASE D, etc.:
#
#     sub-phase X1 | arm1_tr1     on GPU 0 || arm3_tr1     on GPU 1
#     sub-phase X2 | arm5_tr1     on GPU 0 || arm6_v2_tr1  on GPU 1
#     sub-phase X3 | bimoco_tr1   on GPU 0 (solo — odd count of 5)
#
# All 5 arms in a phase finish before the next phase starts (barrier per
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
# Orchestrator progress:       `results/orchestrate_tau_rep.log`.
#
# Usage (typically triggered manually AFTER `orchestrate.sh` completes
# for the base 6-arm sweep — nothing here auto-triggers on that):
#   WT=$HOME/workspaces/contrastive-forecasting \
#     nohup setsid bash orchestrate_tau_rep.sh > orchestrate_tau_rep.log 2>&1 &
#
# Optional env:
#   WAVES="40000 100000 200000"   # override the wave schedule (unusual)
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

FINAL_STEPS=200000

# Wave schedule — one entry per wave / phase. Fields:
#   wave_index | phase_letter | TARGET_STEPS | SAVE_EVERY | EXTRA_SAVES
# Phase letters continue A/B/C from `orchestrate.sh` (base 6-arm sweep),
# so D/E/F are unambiguous in a shared log.
#   phase D (wave 1): 40k,  save-every 10k, extras 2500 + 40000  → _40k.pth
#   phase E (wave 2): 100k, save-every 25k, extra  100000        → _100k.pth
#   phase F (wave 3): 200k, save-every 25k, no extras            → _200k.pth
# Phase F's SAVE_EVERY = 25000 matches the base 6-arm cadence so periodic
# checkpoints line up across all 11 arms.
WAVE_SCHEDULE=(
  "1|D|40000|10000|2500,40000"
  "2|E|100000|25000|100000"
  "3|F|200000|25000|"
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

# Count how many tau_rep arms have a periodic checkpoint at ≥ target_k.
# Used for the end-of-wave summary.
count_arms_at_step(){ # step_k
  local step_k="$1" n=0
  for var in BB_arm1_tr1 BB_arm3_tr1 BB_arm5_tr1 BB_arm6_v2_tr1 BB_bimoco_tr1; do
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

  log "PHASE $letter · sub-phase ${letter}1: arm1_tr1 (GPU 0) + arm3_tr1 (GPU 1)"
  launch_arm arm1_tr1     0 "$letter" "$target" "$se" "$ex" & pid_a=$!
  launch_arm arm3_tr1     1 "$letter" "$target" "$se" "$ex" & pid_b=$!
  wait_pair $pid_a $pid_b "PHASE $letter · sub-phase ${letter}1" \
    || log "PHASE $letter · sub-phase ${letter}1 had failing arm(s) — continuing to ${letter}2"

  log "PHASE $letter · sub-phase ${letter}2: arm5_tr1 (GPU 0) + arm6_v2_tr1 (GPU 1)"
  launch_arm arm5_tr1     0 "$letter" "$target" "$se" "$ex" & pid_a=$!
  launch_arm arm6_v2_tr1  1 "$letter" "$target" "$se" "$ex" & pid_b=$!
  wait_pair $pid_a $pid_b "PHASE $letter · sub-phase ${letter}2" \
    || log "PHASE $letter · sub-phase ${letter}2 had failing arm(s) — continuing to ${letter}3"

  log "PHASE $letter · sub-phase ${letter}3: bimoco_tr1 (GPU 0, solo)"
  launch_arm bimoco_tr1   0 "$letter" "$target" "$se" "$ex"
  local rc_solo=$?
  log "PHASE $letter · sub-phase ${letter}3 done rc=$rc_solo"

  local n_reached
  n_reached=$(count_arms_at_step "$(( target / 1000 ))")
  log "PHASE $letter DONE — wave $wave · arms at ≥ ${target} steps: $n_reached / 5"
}

log "orchestrator start — WT=$WT (tau_rep=1.0 reruns · staged waves)"
log "wave schedule: $(printf '%s ' "${WAVE_SCHEDULE[@]}")"

for entry in "${WAVE_SCHEDULE[@]}"; do
  IFS='|' read -r WAVE LETTER TARGET SAVE_EVERY EXTRAS <<< "$entry"
  run_wave "$WAVE" "$LETTER" "$TARGET" "$SAVE_EVERY" "$EXTRAS"
done

# Summary — count of arms whose FINAL.pth landed (written only after
# wave 3 reaches step FINAL_STEPS).
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
  "expected_backbones": 5,
  "waves": ["40000", "100000", "200000"]
}
EOF
log "state written to $STATE"
