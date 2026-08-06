#!/bin/bash
# #390 — one WAVE of backbone training, run as a GPU slot POOL.
#
# Same contract as `orchestrate.sh` (WAVE / ARMS in, `orchestrate_wave<N>.pid`
# and `orchestrate_wave<N>_state.json` out) with one difference: it keeps
# SLOTS_PER_GPU arms alive on each 4090 and refills a slot the moment one
# frees, instead of joining a whole pair before starting the next. Measured
# on elisa, two trainers per GPU deliver 5.7 sps against 3.2 solo — 1.8x, and
# on a sweep this size that is a day and a half.
#
#   WT=/home/jupyter/wt-cf-390-train WAVE=1 SLOTS_PER_GPU=2 \
#     bash orchestrate_pool.sh
#
# `orchestrate.sh` stays as it is: it is the reviewed, tested, pair-at-a-time
# path, and the tests pin its behaviour.
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"
# shellcheck source=gpu_pool.sh
source "$HERE/gpu_pool.sh"

WAVE="${WAVE:-1}"
case "$WAVE" in
  1|2|3) : ;;
  *) echo "ABORT: WAVE must be 1, 2 or 3; got '$WAVE'" >&2; exit 2 ;;
esac
TARGET_STEPS="${CF390_WAVE_TARGET_STEPS[$WAVE]}"
SAVE_EVERY="${CF390_WAVE_SAVE_EVERY[$WAVE]}"
EXTRA_SAVES="$CF390_EXTRA_SAVES"
FINAL_STEPS="$CF390_FINAL_STEPS"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"

ARMS="${ARMS:-${CF390_ARMS[*]}}"
read -r -a ARM_LIST <<< "$ARMS"

OUT="$WT/experiments/2026-08-01_lalign_teacher"
RES="$OUT/results"; mkdir -p "$RES"
SCRIPTS="$OUT/scripts"
LOG="$RES/orchestrate_wave${WAVE}.log"
STATE="$RES/orchestrate_wave${WAVE}_state.json"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [orch-390-w$WAVE] $*" | tee -a "$LOG"; }

PIDFILE="$RES/orchestrate_wave${WAVE}.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

launch_arm(){ # arm gpu
  local arm="$1" gpu="$2" rc
  log "arm $arm start: BB_GPU=$gpu target=$TARGET_STEPS"
  WT="$WT" BB_GPU="$gpu" TARGET_STEPS="$TARGET_STEPS" \
  FINAL_STEPS="$FINAL_STEPS" SAVE_EVERY="$SAVE_EVERY" EXTRA_SAVES="$EXTRA_SAVES" \
    bash "$SCRIPTS/run_arm.sh" "$arm" >>"$LOG" 2>&1
  rc=$?
  log "arm $arm done rc=$rc"
  return $rc
}

log "orchestrator start — WT=$WT wave=$WAVE target=$TARGET_STEPS arms=${#ARM_LIST[@]} slots/gpu=$SLOTS_PER_GPU"
log "arms: ${ARM_LIST[*]}"

POOL_RC_DIR="$RES/.pool_bb_wave${WAVE}" pool_run "$SLOTS_PER_GPU" launch_arm "${ARM_LIST[@]}"
for arm in "${ARM_LIST[@]}"; do log "rc[$arm]=${POOL_RC[$arm]:-?}"; done

# Summary — how many arms reached this wave's step budget ON DISK. Each arm is
# matched on its exact run name: `arm5` and `arm5_tr1` share the
# `bb_small_arm5` prefix, and a glob that crosses them credits a crashed cell
# with its neighbour's progress. `_FINAL.pth` is counted too, because on the
# last wave the trainer's final write is FINAL, not `_<N>k.pth`.
target_k=$(( TARGET_STEPS / 1000 ))
reached=0
for arm in "${ARM_LIST[@]}"; do
  name="$(bb_name "$arm")" || { log "  $arm: unknown arm — not counted"; continue; }
  best=-1
  for f in "$OUT/runs/${name}"_*k.pth; do
    [ -e "$f" ] || continue
    case "$f" in *_optimizer.pth) continue;; esac
    k=$(basename "$f" | sed -E 's/.*_([0-9]+)k\.pth$/\1/')
    case "$k" in ''|*[!0-9]*) continue;; esac
    (( k > best )) && best=$k
  done
  final="no"
  [ -f "$OUT/runs/${name}_FINAL.pth" ] && { final="yes"; best=$target_k; }
  [ "$best" -ge "$target_k" ] && reached=$((reached + 1))
  log "  $arm: newest checkpoint ${best}k FINAL=$final (target ${target_k}k)"
done
log "orchestrator done — arms at/past ${target_k}k: $reached / ${#ARM_LIST[@]}"

cat > "$STATE" <<EOF
{
  "state": "done",
  "wave": $WAVE,
  "target_steps": $TARGET_STEPS,
  "slots_per_gpu": $SLOTS_PER_GPU,
  "arms_reached_target": $reached,
  "arms_expected": ${#ARM_LIST[@]}
}
EOF
log "state written to $STATE"
