#!/bin/bash
# #379 base-fresh orchestrator — STAGED WAVES for the 2 base arms that
# never started (arm6_v2, bimoco).
#
# The base-6 orchestrator (`orchestrate.sh`) trains all six arms
# end-to-end. Two of those runs (arm6_v2, bimoco) hadn't launched by the
# time the researcher decided to add the tau_rep / no_sigreg_e / no_cpc
# variants and impose a Wave-D-first barrier. This script re-uses the
# `run_arm.sh` wave contract to bring those two remaining base arms
# through the same 40k → 100k → 200k schedule as the variant orchestrators
# so all 23 arms hit step 40 000 first before the researcher decides who
# advances.
#
# Phase letters M/N/O continue the shared-log lettering after tau_rep
# (D/E/F), nse (G/H/I) and ncpc (J/K/L).
#
#   PHASE M (wave 1) → step  40 000 (save-every 10 000, extras {2500, 40000})
#   PHASE N (wave 2) → step 100 000 (save-every 25 000, extra  {100000})
#   PHASE O (wave 3) → step 200 000 (save-every 25 000, no extras)
#
# Only two arms → a single sub-phase per wave (letters follow the parent
# PHASE — M1 belongs to PHASE M, etc.):
#
#     sub-phase X1 | arm6_v2 on GPU 0 || bimoco on GPU 1
#
# `run_arm.sh` accepts TARGET_STEPS / FINAL_STEPS: only the final wave
# writes `_FINAL.pth`; intermediate waves leave the `_<N>k.pth` on disk
# so the next wave's launcher resumes from it via `--resume`.
#
# Per-arm training log stream: `results/dl_<arm>.log`.
# Orchestrator progress:       `results/orchestrate_base_fresh.log`.
#
# Usage:
#   WT=$HOME/workspaces/contrastive-forecasting \
#     nohup setsid bash orchestrate_base_fresh.sh > orchestrate_base_fresh.log 2>&1 &
#
# Optional env:
#   MAX_WAVE=M          # break after PHASE M (Wave-D barrier: only reach 40k)
#                       #   unset → run all waves (M → N → O)
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
LOG="$RES/orchestrate_base_fresh.log"
STATE="$RES/orchestrate_base_fresh_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-379-bf] $*" | tee -a "$LOG"; }

# Backbone base names — must match run_arm.sh's case block.
BB_arm6_v2="bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"
BB_bimoco="bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090"

FINAL_STEPS=200000

# Optional early-exit letter (Wave-D barrier). Unset → run every wave.
MAX_WAVE="${MAX_WAVE:-}"

WAVE_SCHEDULE=(
  "1|M|40000|10000|2500,40000"
  "2|N|100000|25000|100000"
  "3|O|200000|25000|"
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

count_arms_at_step(){ # step_k
  local step_k="$1" n=0
  for var in BB_arm6_v2 BB_bimoco; do
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

  log "PHASE $letter · sub-phase ${letter}1: arm6_v2 (GPU 0) + bimoco (GPU 1)"
  launch_arm arm6_v2 0 "$letter" "$target" "$se" "$ex" & pid_a=$!
  launch_arm bimoco  1 "$letter" "$target" "$se" "$ex" & pid_b=$!
  wait_pair $pid_a $pid_b "PHASE $letter · sub-phase ${letter}1" \
    || log "PHASE $letter · sub-phase ${letter}1 had failing arm(s)"

  local n_reached
  n_reached=$(count_arms_at_step "$(( target / 1000 ))")
  log "PHASE $letter DONE — wave $wave · arms at ≥ ${target} steps: $n_reached / 2"
}

log "orchestrator start — WT=$WT (base-fresh arm6_v2 + bimoco · staged waves · MAX_WAVE='${MAX_WAVE}')"
log "wave schedule: $(printf '%s ' "${WAVE_SCHEDULE[@]}")"

for entry in "${WAVE_SCHEDULE[@]}"; do
  IFS='|' read -r WAVE LETTER TARGET SAVE_EVERY EXTRAS <<< "$entry"
  run_wave "$WAVE" "$LETTER" "$TARGET" "$SAVE_EVERY" "$EXTRAS"
  if [ -n "$MAX_WAVE" ] && [ "$LETTER" = "$MAX_WAVE" ]; then
    log "MAX_WAVE=$MAX_WAVE reached — stopping after PHASE $LETTER"
    break
  fi
done

count_final=0
for var in BB_arm6_v2 BB_bimoco; do
  bb="${!var}"
  [ -f "$OUT/runs/${bb}_FINAL.pth" ] && count_final=$((count_final + 1))
done
log "orchestrator done — base-fresh backbones FINAL: $count_final / 2"

cat > "$STATE" <<EOF
{
  "state": "done",
  "backbones_final": $count_final,
  "expected_backbones": 2,
  "waves": ["40000", "100000", "200000"],
  "max_wave": "${MAX_WAVE}"
}
EOF
log "state written to $STATE"
