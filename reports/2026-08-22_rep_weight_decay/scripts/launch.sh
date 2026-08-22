#!/bin/bash
# #409 — the entry point. Eight arms, two cards, one command.
#
# elisa holds two RTX 4090s. This deals the eight arms round-robin over them
# and starts one lane on each (`phase1.sh`). Each lane trains its arms in turn:
# backbone to 40,000 steps, then a 30,000-step student head, then that head's
# 97 GIFT-Eval configs. Every arm is independent, so the deal is free.
#
# Everything is idempotent. Re-run this after a reboot: a stop whose checkpoint
# is on disk is a no-op, a leg resumes the cell's furthest checkpoint with its
# optimizer state, and a scored tag skips its head and its eval.
#
# ---- What keeps the study alive ----------------------------------------------
#
#   a crashed leg   `phase1.sh` re-fires it CF409_LEG_TRIES times.
#   a lost arm      `auc_guard.sh` stops a leg that lost the contrastive task
#                   and the lane moves to the next arm. No re-fire: a re-fire
#                   trains the same collapse.
#   a dead session  this loop rewrites results/RUN_STATE.md and re-runs
#                   collect.sh every STATE_EVERY seconds, so a session that
#                   picks the study up reads one file and sees the table.
#
# Usage:
#   nohup setsid bash scripts/launch.sh >/dev/null 2>&1 &
#
#   GPUS="0" bash scripts/launch.sh          # one card
#   ARMS="dec0_s20 ctrl_s20" bash scripts/launch.sh
#   CF409_TRIAL=400 bash scripts/launch.sh   # the whole pipeline, in minutes
#   CF409_DRY_RUN=1 bash scripts/launch.sh   # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

GPUS="${GPUS:-$(cf409_default_gpus)}"
ARMS="${ARMS:-$CF409_ARMS}"
# Every lane opens the same streaming dataset. Starting them together puts two
# cold HF readers on one connection.
STAGGER="${STAGGER:-180}"
STATE_EVERY="${STATE_EVERY:-1800}"
mkdir -p "$CF409_RESULTS" "$CF409_PLOTS"

LOG="$CF409_RESULTS/launch.log"
STATE="$CF409_RESULTS/RUN_STATE.md"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#409] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$GPUS"
read -r -a arm_list <<<"$ARMS"
[ "${#gpu_list[@]}" -ge 1 ] || { echo "ABORT: GPUS is empty" >&2; exit 2; }
cf409_require_gpus "$GPUS" || exit 2
for arm in "${arm_list[@]}"; do cf409_require_arm "$arm" || exit $?; done

lane_of(){  # <arm index>
  echo $(( $1 % ${#gpu_list[@]} ))
}

if [ -n "${CF409_DRY_RUN:-}" ]; then
  echo "study cell=$CF409_CELL k=$CF409_K reduce=$CF409_REDUCE" \
       "stops='$CF409_STOPS' head=$CF409_HEAD_STEPS"
  echo "  root=$CF409_ROOT results=$CF409_RESULTS gpus='$GPUS'"
  for i in "${!arm_list[@]}"; do
    echo "arm ${arm_list[$i]} gpu=${gpu_list[$(lane_of "$i")]}" \
         "rep_end=$(cf409_rep_end "${arm_list[$i]}")" \
         "seed=$(cf409_seed "${arm_list[$i]}")" \
         "target=$(cf409_align_target "${arm_list[$i]}")"
  done
  exit 0
fi

# The checkout is checked before the first leg, not after eight of them. A
# stale one trains eight copies of the control and says nothing unusual. It
# runs after the plan print, so a dry run works from any checkout — a worktree
# carries no HF token, and the token is one of the three things checked.
cf409_check_checkout || exit 6

# What a re-dispatched session reads first. One file, overwritten, so it is
# never a log to scroll.
state(){  # <note>
  { echo "# #409 run state — the L_rep weight decay at k = 3"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- note: $1"
    echo "- cell: \`$CF409_CELL\`, k = $CF409_K, reduce \`$CF409_REDUCE\`"
    echo "- arms: $ARMS"
    echo "- cards: $GPUS, launcher pid $$"
    echo "- root: \`$CF409_ROOT\`"
    echo
    echo "## Scores"
    echo
    echo '```'
    cat "$CF409_RESULTS/scores.csv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "## Contrastive AUC"
    echo
    echo '```'
    cat "$CF409_RESULTS/auc_verdicts.tsv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "## Backbones on disk"
    echo
    echo '```'
    ls -1 "$CF409_ROOT"/*/*/leg_*k/*k.pth 2>/dev/null \
      | grep -v optimizer | sed "s#$CF409_ROOT/##" || echo "(none yet)"
    echo '```'
  } >"$STATE.tmp" && mv -f "$STATE.tmp" "$STATE"
}

log "START arms='$ARMS' gpus='$GPUS' root=$CF409_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  2>/dev/null | sed 's/^/  gpu /' | tee -a "$LOG"
state "starting"

pids=(); names=()
for lane in "${!gpu_list[@]}"; do
  gpu="${gpu_list[$lane]}"
  lane_arms=""
  for i in "${!arm_list[@]}"; do
    [ "$(lane_of "$i")" = "$lane" ] && lane_arms="$lane_arms ${arm_list[$i]}"
  done
  [ -n "$lane_arms" ] || continue
  log "gpu $gpu takes${lane_arms}"
  ARMS="${lane_arms# }" BB_GPU="$gpu" \
    nohup bash "$HERE/phase1.sh" \
      >>"$CF409_RESULTS/phase1_gpu${gpu}.out" 2>&1 &
  pids+=($!); names+=("gpu=$gpu arms=${lane_arms# }")
  sleep "$STAGGER"
done

# The reporting half. It refreshes the table while the lanes work, and it stops
# when they do.
running(){
  local p
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done
  return 1
}
while running; do
  bash "$HERE/collect.sh" >>"$LOG" 2>&1
  state "${#pids[@]} lane(s) running"
  sleep "$STATE_EVERY"
done

failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  log "lane ${names[$i]} rc=$rc"
  [ $rc -eq 0 ] || failed=$(( failed + 1 ))
done
bash "$HERE/collect.sh" 2>&1 | tee -a "$LOG"
state "done — $failed lane(s) failed"
log "DONE — $failed lane(s) failed"
[ "$failed" -eq 0 ] || exit 1
