#!/bin/bash
# #390 — the measurement half of one wave: q-head + GIFT-Eval for every arm
# in the wave, two at a time (one per GPU).
#
#   WT=/home/jupyter/wt-cf-390-train WAVE=1 bash eval_wave.sh
#   WAVE=3 ARMS="arm5_ncpc arm6_v2_tr1" bash eval_wave.sh
#
# The wave sets the pair (backbone step, head steps) that `eval_arm.sh` needs:
#
#   wave 1 |  40k backbone · 15 000-step head
#   wave 2 | 100k backbone · 30 000-step head
#   wave 3 | 200k backbone · 30 000-step head
#
# An arm that fails does not stop the wave — the cells are independent. The
# per-arm exit status lands in the log and in the wave's status file, so the
# pipeline can tell a finished wave from a wave with a hole in it.
set -uo pipefail

WT="${WT:-$HOME/wt-cf-390-train}"
case "$WT" in
  /tmp/*|/tmp) echo "ABORT: WT=$WT is under /tmp — refusing." >&2; exit 2 ;;
esac

# `cd -P` for the same reason as eval_arm.sh: this file is reached through a
# `scripts/` symlink inside $WT, and the shared libraries must come from this
# checkout rather than from whatever commit $WT sits on.
HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -P "$HERE/../../.." && pwd)"
# shellcheck source=arm_names.sh
source "$HERE/arm_names.sh"
# shellcheck source=gpu_pool.sh
source "$HERE/gpu_pool.sh"
# shellcheck source=/dev/null
source "$ROOT/scripts/eval_cell_identity.sh"

SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
# The head seed the wave runs. `eval_arm.sh` reads it from the environment
# and puts it in every cell name it writes, so a wave that did not bind it
# here counted its results at the library's default while writing them under
# whatever seed happened to be exported: ten measured cells, ten MISSING
# lines. One variable, handed to the stage and used for the lookup.
HEAD_SEED="${HEAD_SEED:-$EVAL_DEFAULT_HEAD_SEED}"
WAVE="${WAVE:-1}"
case "$WAVE" in
  1) BB_STEP_K=40;  HEAD_STEPS=15000 ;;
  2) BB_STEP_K=100; HEAD_STEPS=30000 ;;
  3) BB_STEP_K=200; HEAD_STEPS=30000 ;;
  *) echo "ABORT: WAVE must be 1, 2 or 3; got '$WAVE'" >&2; exit 2 ;;
esac

ARMS="${ARMS:-${CF390_ARMS[*]}}"
read -r -a ARM_LIST <<< "$ARMS"

EXP="$WT/experiments/2026-08-01_lalign_teacher"
RES="$EXP/results"; mkdir -p "$RES"
SCRIPTS="$EXP/scripts"
LOG="$RES/eval_wave${WAVE}.log"
STATUS="$RES/eval_wave${WAVE}_status.tsv"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [eval-390-w$WAVE] $*" | tee -a "$LOG"; }

PIDFILE="$RES/eval_wave${WAVE}.pid"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

run_one(){ # arm gpu
  local arm="$1" gpu="$2" rc
  WT="$WT" ARM="$arm" BB_GPU="$gpu" BB_STEP_K="$BB_STEP_K" HEAD_STEPS="$HEAD_STEPS" \
    HEAD_SEED="$HEAD_SEED" \
    bash "$SCRIPTS/eval_arm.sh" >>"$LOG" 2>&1
  rc=$?
  log "arm $arm eval rc=$rc"
  return $rc
}

log "eval wave start — WT=$WT bb=${BB_STEP_K}k head=${HEAD_STEPS} head_seed=${HEAD_SEED} arms=${#ARM_LIST[@]} slots/gpu=$SLOTS_PER_GPU"
log "arms: ${ARM_LIST[*]}"
: > "$STATUS"

POOL_RC_DIR="$RES/.pool_eval_wave${WAVE}" pool_run "$SLOTS_PER_GPU" run_one "${ARM_LIST[@]}"
for arm in "${ARM_LIST[@]}"; do
  printf '%s\t%s\n' "$arm" "${POOL_RC[$arm]:-?}" >> "$STATUS"
done

# Summary straight off disk: a cell counts as measured only when its
# `<cell>_summary.txt` exists AND says 97 configs. eval_arm.sh refuses to
# write that file on a partial, so this is the honest count. The cell name
# carries the replicate its backbone came from and the seed its head was
# trained under; the wave-2 and wave-3 backbones here are all resumes, so
# the lookup asks for either replicate — at the seed this wave just ran.
ok=0
for arm in "${ARM_LIST[@]}"; do
  mapfile -t found < <(eval_cell_summaries "$EXP/eval_gm_mase" "$arm" \
                         "$BB_STEP_K" "$HEAD_STEPS" "$HEAD_SEED")
  # What was looked for, named the way the cells are.
  cell="$(eval_cell_name "$arm" "$BB_STEP_K" "[_r<N>]" "$HEAD_STEPS" "$HEAD_SEED")"
  if [ "${#found[@]}" -gt 1 ]; then
    log "  $cell: AMBIGUOUS — ${#found[@]} replicate cells measured, not counting one:"
    for f in "${found[@]}"; do log "    $(basename "$f")"; done
  elif [ "${#found[@]}" -eq 1 ] && grep -q "(97 configs)" "${found[0]}"; then
    ok=$((ok + 1)); log "  $(basename "${found[0]}" _summary.txt): $(head -1 "${found[0]}")"
  else
    log "  $cell: MISSING or partial"
  fi
done
log "eval wave done — cells with a full 97-config measurement: $ok / ${#ARM_LIST[@]}"
