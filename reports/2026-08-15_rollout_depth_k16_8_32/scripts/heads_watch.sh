#!/bin/bash
# #401 — on elisa: one head and one 97-config GIFT-Eval per backbone stop, as
# each stop lands.
#
# The backbones train on a rented box and arrive here through the sync loop,
# 15 minutes apart (`sync/launch_sync.sh`). This watcher polls for stops whose
# checkpoint is here and whose score file is not, and fires `head_eval.sh` on
# each. It replaces nothing: same head trainer, same 30,000-step phase-1
# budget, same eval, same score file.
#
# Why a watcher and not phase1.sh. `phase1.sh` trains the backbone itself and
# then its head. Here the backbone trains somewhere else, so the loop that
# waits for it has to be its own process — one that survives a leg that
# arrives six hours after the one before it.
#
# Phase order. Every phase-1 head (30,000 steps) runs before any phase-2 head
# (40k / 100k / 200k steps), because phase 1 is what picks the arms and
# phase 2 costs up to seven times as much per head. Phase 2 starts by itself
# once every phase-1 cell of the pass holds a score.
#
# The root. Unset, it is the box's tree where the sync loop lands it
# (CF401_SYNC_ROOT). The box owns both backbone arms, so on elisa the root is
# never a local checkpoints directory — see study.sh, "Which machine owns which
# arm".
#
# Concurrency. One head per GPU, through `head_eval_bb.sh`'s own VRAM gate and
# its per-GPU lock. HEAD_GPUS is the card list, and it is `0`: elisa GPU 1
# belongs to another session and this study does not take it. The GIFT-Eval
# that follows each head runs on the CPU, so several evals overlap without
# touching a card.
#
# Everything is idempotent. A scored tag is a no-op, a trained head skips
# straight to its eval, and an eval resumes per shard.
#
# Usage:
#   HEAD_GPUS="0" nohup setsid bash scripts/heads_watch.sh &
#
#   ONCE=1 bash scripts/heads_watch.sh          # one pass, then exit
#   CF401_DRY_RUN=1 bash scripts/heads_watch.sh # print the plan, run nothing
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CF401_ROOT_GIVEN="${CF401_ROOT:-}"
. "$HERE/study.sh"
cf401_use_root "$CF401_SYNC_ROOT"

HEAD_GPUS="${HEAD_GPUS:-0}"
POLL="${POLL:-300}"
ONCE="${ONCE:-}"
mkdir -p "$CF401_RESULTS"

LOG="$CF401_RESULTS/heads_watch.log"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 heads] $*" | tee -a "$LOG"; }

read -r -a gpu_list <<<"$HEAD_GPUS"
[ "${#gpu_list[@]}" -ge 1 ] || { echo "ABORT: HEAD_GPUS is empty" >&2; exit 2; }

# This watcher reads ONE root. A backbone arm of the same reduction that runs
# on this machine under ANOTHER root is therefore an arm nothing here will ever
# score, and a leg that climbs for 33 h unread is the most expensive failure
# this card has. The layout has one tree for the backbones — the box's — so
# this case is a mistake, not a configuration.
#
# It also takes the card. The heads run on elisa GPU 0, and a local backbone
# arm holds that card, so every head of phase 1 would wait on
# `head_eval_bb.sh`'s VRAM gate and time out.
#
# CF401_ALLOW_LOCAL_ARMS=1 waives it, for a session that means to score one
# root while another climbs.
guard_local_arms(){
  local hit
  [ -z "${CF401_ALLOW_LOCAL_ARMS:-}" ] || return 0
  hit="$(cf401_running_legs | cf401_foreign_arm "$CF401_REDUCE" "$CF401_ROOT")" \
    || return 0
  echo "ABORT: a $CF401_REDUCE backbone arm runs on this machine under" >&2
  echo "  another root. pid, reduction, root:" >&2
  printf '  %s\n' "$hit" >&2
  echo "  This watcher reads $CF401_ROOT, so those arms are never scored," >&2
  echo "  and they hold the card the heads need." >&2
  echo "  Stop them, point CF401_ROOT at their root, or set" >&2
  echo "  CF401_ALLOW_LOCAL_ARMS=1 to score this root alone." >&2
  return 3
}
guard_local_arms || exit $?

# The score file one (depth, stop, head budget) writes, through the same tag
# builder head_eval.sh and collect.sh use.
score_file(){  # <k> <stop> <head steps>
  printf '%s/score_%s.txt\n' "$CF401_RESULTS" "$(cf401_tag "$1" "$2" "$3")"
}

# A cell is due when its backbone is here and its score file is not. An empty
# score file counts as missing: an eval killed between opening and writing
# leaves one, and 0.0 would be the best GM-Relative MASE this project ever
# recorded.
# Is this cell already training or evaluating, in ANY process?
#
# `cell_is_due` reads the score file, and a cell that still runs has not
# written one. With one slot per card that never showed. With two slots it
# does: at 17:21 on 2026-08-19 `k32_bb100k_h100k` was 4 h into slot 1, its
# score file was absent, and the free slot 0 started the same cell again. The
# duplicate then blocked on the per-card lock and held a slot that the two
# 200,000-step cells needed.
#
# This asks the process table, not the slot map, because a RESTARTED watcher
# has empty slots and would repeat the same mistake against the heads its
# predecessor left running.
cell_is_running(){  # <k> <stop> <head steps>
  local want="head_eval.sh $1 $2 $3" p cmd
  for p in $(pgrep -f 'head_eval\.sh' 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    # `/proc/<pid>/cmdline` and not the `pgrep` pattern, because `pgrep -f`
    # also matches a process that only NAMES the file, this check among them.
    # `test_the_sync_check_reads_the_argument_list` holds study.sh to the same
    # rule.
    cmd="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)"
    case "$cmd" in *"$want "*) ;; *) continue ;; esac
    # Same study instance, or none. A test points CF401_ROOT at a temporary
    # tree and a real head carries this study's root, so without this the
    # dry run of a test would drop whatever cells happen to train on the
    # machine at that moment, and the test would read live state.
    grep -qz "CF401_ROOT=$CF401_ROOT" "/proc/$p/environ" 2>/dev/null && return 0
  done
  return 1
}

cell_is_due(){  # <k> <stop> <head steps>
  local bb
  [ -s "$(score_file "$1" "$2" "$3")" ] && return 1
  cell_is_running "$1" "$2" "$3" && return 1
  bb="$(cf401_bb_ckpt "$1" "$2")"
  [ -n "$bb" ] && [ -f "$bb" ]
}

# Phase 1 covers every (depth, stop) at the fixed budget. Phase 2 repeats the
# two best arms with the head budget set to the backbone stop.
phase1_jobs(){
  local k stop
  for k in $CF401_DEPTHS; do
    for stop in $CF401_STOPS; do
      printf '%s %s %s\n' "$k" "$stop" "$CF401_HEAD_STEPS_P1"
    done
  done
}

phase1_is_complete(){
  local k stop
  for k in $CF401_DEPTHS; do
    for stop in $CF401_STOPS; do
      [ -s "$(score_file "$k" "$stop" "$CF401_HEAD_STEPS_P1")" ] || return 1
    done
  done
  return 0
}

# The two arms phase 2 repeats. `pick_phase2_arms.py` refuses an incomplete
# phase 1, which is the card's own rule. ARMS overrides it.
phase2_arms(){
  if [ -n "${ARMS:-}" ]; then printf '%s\n' "$ARMS"; return 0; fi
  python3 "$HERE/pick_phase2_arms.py" --scores "$CF401_RESULTS/scores.csv" \
    2>>"$LOG"
}

phase2_jobs(){
  local arms k stop
  arms="$(phase2_arms)" || return 1
  for k in $arms; do
    cf401_is_in "$k" "$CF401_DEPTHS" || continue
    for stop in $CF401_STOPS; do
      # The card's rule: head steps = backbone steps.
      printf '%s %s %s\n' "$k" "$stop" "$stop"
    done
  done
}

# Every job of this pass, phase 1 first. Phase 2 is asked for only once every
# phase-1 cell holds a score, so the picker never sees a table with a hole.
due_jobs(){
  local k stop steps
  while read -r k stop steps; do
    [ -n "$k" ] || continue
    cell_is_due "$k" "$stop" "$steps" && printf '%s %s %s\n' "$k" "$stop" "$steps"
  done < <(phase1_jobs)
  phase1_is_complete || return 0
  while read -r k stop steps; do
    [ -n "$k" ] || continue
    cell_is_due "$k" "$stop" "$steps" && printf '%s %s %s\n' "$k" "$stop" "$steps"
  done < <(phase2_jobs)
}

if [ -n "${CF401_DRY_RUN:-}" ]; then
  echo "watch reduce=$CF401_REDUCE root=$CF401_ROOT results=$CF401_RESULTS"
  echo "  gpus='$HEAD_GPUS' poll=${POLL}s"
  bash "$HERE/collect.sh" >/dev/null 2>&1
  due_jobs | while read -r k stop steps; do
    echo "head k=$k stop=$stop steps=$steps enc=$CF401_ENC"
  done
  exit 0
fi

log "START reduce=$CF401_REDUCE gpus='$HEAD_GPUS' poll=${POLL}s root=$CF401_ROOT"

# One slot per entry in HEAD_GPUS. The entry names the card the slot's head
# trains on, and the SAME card can appear twice.
#
# A cell is a head train on the GPU and then a 97-config eval on the CPUs, and
# `head_eval_bb.sh` closes its GPU lock and drops its VRAM before that eval.
# So one slot per card leaves the card idle for the whole eval, which is about
# 1.2 h of every cell. `HEAD_GPUS="0 0"` gives card 0 two slots: the second
# head trains while the first cell evaluates. The per-card lock inside
# `head_eval_bb.sh` still lets only one of them hold the card.
declare -A slot_pid=() slot_name=()

reap_slots(){  # <block: 1 waits for one to finish>
  local i rc freed=0
  for i in "${!gpu_list[@]}"; do
    [ -n "${slot_pid[$i]:-}" ] || { freed=1; continue; }
    if ! kill -0 "${slot_pid[$i]}" 2>/dev/null; then
      wait "${slot_pid[$i]}"; rc=$?
      log "head ${slot_name[$i]} rc=$rc"
      unset 'slot_pid[$i]'; freed=1
    fi
  done
  [ "$freed" = "1" ] || [ "${1:-0}" != "1" ] || sleep 20
}

free_slot(){
  local i
  for i in "${!gpu_list[@]}"; do
    [ -n "${slot_pid[$i]:-}" ] || { printf '%s\n' "$i"; return 0; }
  done
  return 1
}

while true; do
  reap_slots
  # `collect.sh` rebuilds scores.csv and splits.csv from the score files, and
  # it runs BEFORE the job list, because the phase-2 half of that list is what
  # `pick_phase2_arms.py` reads out of scores.csv.
  bash "$HERE/collect.sh" >>"$LOG" 2>&1
  started=0
  while read -r k stop steps; do
    [ -n "$k" ] || continue
    slot="$(free_slot)" || break
    gpu="${gpu_list[$slot]}"
    tag="$(cf401_tag "$k" "$stop" "$steps")"
    log "head $tag START on gpu $gpu slot $slot ($(basename "$(cf401_bb_ckpt "$k" "$stop")"))"
    BB_GPU="$gpu" GPU_GATE_LOCKDIR="${GPU_GATE_LOCKDIR:-/tmp}" \
      nohup bash "$HERE/head_eval.sh" "$k" "$stop" "$steps" \
        >>"$CF401_RESULTS/head_${tag}.out" 2>&1 &
    slot_pid[$slot]=$!; slot_name[$slot]="$tag"
    started=$(( started + 1 ))
  done < <(due_jobs)

  if [ -n "$ONCE" ]; then
    for i in "${!gpu_list[@]}"; do
      [ -n "${slot_pid[$i]:-}" ] && { wait "${slot_pid[$i]}"
                                      log "head ${slot_name[$i]} rc=$?"; }
    done
    log "ONCE pass done — $started head(s) started"
    exit 0
  fi

  # Nothing due and nothing running means every stop that is here is scored.
  # The loop keeps polling, because the box is still climbing to the next one.
  [ "$started" -gt 0 ] || sleep "$POLL"
  reap_slots 1
done
