#!/bin/bash
# #390 orchestrator — one WAVE of the 10 L_align cells across elisa's two
# 4090s. The issue's schedule is three waves:
#
#   wave 1 | backbone 40 000   →  15 000-step q-head  →  GIFT-Eval (97 cfgs)
#   wave 2 | backbone 100 000  →  30 000-step q-head  →  GIFT-Eval
#   wave 3 | backbone 200 000, only the cells whose wave-2 GM-Relative MASE
#            fell below their wave-1 value
#
# This script runs the BACKBONE half of one wave. Head training and
# GIFT-Eval are separate stages (see the parent #379 experiment's
# eval_2L_gm_mase.sh). Invoke it once per wave:
#
#   WT=$HOME/workspaces/contrastive-forecasting WAVE=1 \
#     nohup setsid bash orchestrate.sh > /dev/null 2>&1 &
#
# WAVE ∈ {1,2,3} selects TARGET_STEPS / SAVE_EVERY. ARMS can be overridden
# to restrict wave 3 to the cells that kept improving:
#
#   WAVE=3 ARMS="arm5_ncpc arm6_v2_tr1" bash orchestrate.sh
#
# `run_arm.sh` is idempotent twice over: a completed arm short-circuits on
# `_FINAL.pth`, and an intermediate wave short-circuits when a `_<N>k.pth`
# at or past TARGET_STEPS already exists. Re-running after a crash resumes
# from whatever is on disk.
#
# Pairs run two at a time (one per GPU). 10 arms → 5 phases per wave.
set -uo pipefail

WT="${WT:-$HOME/workspaces/contrastive-forecasting}"
case "$WT" in
  /tmp/*|/tmp)
    echo "ABORT: WT=$WT is under /tmp — refusing." >&2
    exit 2
    ;;
esac

WAVE="${WAVE:-1}"
case "$WAVE" in
  1) TARGET_STEPS=40000;  SAVE_EVERY=10000; EXTRA_SAVES="2500" ;;
  2) TARGET_STEPS=100000; SAVE_EVERY=25000; EXTRA_SAVES="2500" ;;
  3) TARGET_STEPS=200000; SAVE_EVERY=25000; EXTRA_SAVES="2500" ;;
  *) echo "ABORT: WAVE must be 1, 2 or 3; got '$WAVE'" >&2; exit 2 ;;
esac
# FINAL_STEPS stays at the arm's true end (200k) for waves 1 and 2, so
# run_arm.sh writes no `_FINAL.pth` and the next wave resumes from the
# latest `_<N>k.pth`. See run_arm.sh § Staged-wave support.
FINAL_STEPS=200000

# All 10 cells by default. Wave 3 is expected to carry a subset.
ARMS="${ARMS:-arm5 arm5_tr1 arm5_nse arm5_ncpc arm5_combab arm6_v2 arm6_v2_tr1 arm6_v2_nse arm6_v2_ncpc arm6_v2_combab}"
read -r -a ARM_LIST <<< "$ARMS"

OUT="$WT/experiments/2026-08-01_lalign_teacher"
RES="$OUT/results"; mkdir -p "$RES"
SCRIPTS="$OUT/scripts"
LOG="$RES/orchestrate_wave${WAVE}.log"
STATE="$RES/orchestrate_wave${WAVE}_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-390-w$WAVE] $*" | tee -a "$LOG"; }

launch_arm(){ # arm bb_gpu
  local arm="$1" bb_gpu="$2"
  log "arm $arm start: BB_GPU=$bb_gpu target=$TARGET_STEPS"
  WT="$WT" BB_GPU="$bb_gpu" TARGET_STEPS="$TARGET_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" SAVE_EVERY="$SAVE_EVERY" EXTRA_SAVES="$EXTRA_SAVES" \
    bash "$SCRIPTS/run_arm.sh" "$arm" >>"$LOG" 2>&1
  local rc=$?
  log "arm $arm done rc=$rc"
  return $rc
}

log "orchestrator start — WT=$WT wave=$WAVE target=$TARGET_STEPS arms=${#ARM_LIST[@]}"
log "arms: ${ARM_LIST[*]}"

# Two at a time, one per GPU. An arm that fails does not stop the sweep —
# the cells are independent, and a dropped cell is the issue's stop rule.
i=0
while [ "$i" -lt "${#ARM_LIST[@]}" ]; do
  a="${ARM_LIST[$i]}"
  b="${ARM_LIST[$((i + 1))]:-}"
  if [ -n "$b" ]; then
    log "PHASE $((i / 2 + 1)): $a on GPU 0, $b on GPU 1"
    launch_arm "$a" 0 & pid_a=$!
    launch_arm "$b" 1 & pid_b=$!
    wait "$pid_a"; ra=$?
    wait "$pid_b"; rb=$?
    log "phase $((i / 2 + 1)) joined: rc=$ra/$rb"
  else
    log "PHASE $((i / 2 + 1)): $a on GPU 0 (odd arm count — solo)"
    launch_arm "$a" 0; ra=$?
    log "phase $((i / 2 + 1)) joined: rc=$ra"
  fi
  i=$((i + 2))
done

# Summary — how many arms reached this wave's step budget on disk.
target_k=$(( TARGET_STEPS / 1000 ))
reached=0
for arm in "${ARM_LIST[@]}"; do
  best=-1
  for f in "$OUT/runs/"*"_${arm}_"*k.pth "$OUT/runs/"*"${arm}"*_*k.pth; do
    [ -e "$f" ] || continue
    case "$f" in *_optimizer.pth) continue;; esac
    k=$(basename "$f" | sed -E 's/.*_([0-9]+)k\.pth$/\1/')
    case "$k" in ''|*[!0-9]*) continue;; esac
    (( k > best )) && best=$k
  done
  [ "$best" -ge "$target_k" ] && reached=$((reached + 1))
  log "  $arm: newest checkpoint ${best}k (target ${target_k}k)"
done
log "orchestrator done — arms at/past ${target_k}k: $reached / ${#ARM_LIST[@]}"

cat > "$STATE" <<EOF
{
  "state": "done",
  "wave": $WAVE,
  "target_steps": $TARGET_STEPS,
  "arms_reached_target": $reached,
  "arms_expected": ${#ARM_LIST[@]}
}
EOF
log "state written to $STATE"
