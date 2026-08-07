#!/bin/bash
# #393 — the cap on how many GIFT-Evals elisa runs at once.
#
# Source this and call `eval_slot` before starting an eval. It returns
# once a slot is free and holds it until the calling shell exits.
#
# Why a cap at all: elisa is shared with other agent sessions, and the
# eval is CPU-bound — one process pins one core for hours. Without a cap
# the broker would start one eval per finished head across seven machines
# and take the box's whole core budget.
#
# The cap is on CONCURRENT EVALS, not on worker processes. Each eval runs
# EVAL_SHARDS shard processes, so the core count is
# CF393_EVAL_SLOTS * EVAL_SHARDS. The default 3 x 4 = 12 of elisa's 32
# cores, leaving 20 free against the eight the brief requires. GPU
# headroom is total rather than partial: eval_local.sh runs `--device cpu`,
# so the evals take no VRAM and no SM time, and both 4090s stay with the
# cells that need them.
#
# A counting semaphore, not one lock: N lock files, and a caller takes the
# first it can `flock -n`. flock is held by a file descriptor, so a slot is
# released when its holder exits however it exits — including kill -9,
# which a counter file would not survive.

CF393_EVAL_SLOTS="${CF393_EVAL_SLOTS:-3}"
CF393_EVAL_SLOTDIR="${CF393_EVAL_SLOTDIR:-/tmp/cf393_evalslots}"
CF393_EVAL_SLOT_POLL="${CF393_EVAL_SLOT_POLL:-20}"
# A 97-config eval is ~3 core-hours split over the shards. The wait can be
# long when the queue is deep; timing out into a failure would lose a head
# that already cost GPU time, so the ceiling is a day.
CF393_EVAL_SLOT_TIMEOUT="${CF393_EVAL_SLOT_TIMEOUT:-86400}"

eval_slot_log() { echo "[$(date '+%m-%d %H:%M:%S')] [eval-slot] $*"; }

# Block until one of CF393_EVAL_SLOTS is free. 0 on success, 1 on timeout.
# The slot index is left in CF393_EVAL_SLOT_HELD for the caller's log.
eval_slot() {
  local i waited=0
  mkdir -p "$CF393_EVAL_SLOTDIR" 2>/dev/null || {
    eval_slot_log "cannot create $CF393_EVAL_SLOTDIR; proceeding uncapped"
    return 0; }

  while :; do
    for (( i = 0; i < CF393_EVAL_SLOTS; i++ )); do
      # fd 8 stays open for the life of this shell: the slot is the
      # process, not a file the process has to remember to delete.
      exec 8>>"$CF393_EVAL_SLOTDIR/slot_$i" || continue
      if flock -n 8; then
        CF393_EVAL_SLOT_HELD="$i"
        [ "$waited" -gt 0 ] && eval_slot_log "slot $i after ${waited}s"
        return 0
      fi
      exec 8>&-
    done
    if [ "$waited" -ge "$CF393_EVAL_SLOT_TIMEOUT" ]; then
      eval_slot_log "TIMEOUT after ${waited}s; all $CF393_EVAL_SLOTS slots busy"
      return 1
    fi
    [ $(( waited % 300 )) -eq 0 ] && \
      eval_slot_log "all $CF393_EVAL_SLOTS slots busy, ${waited}s so far"
    sleep "$CF393_EVAL_SLOT_POLL"
    waited=$(( waited + CF393_EVAL_SLOT_POLL ))
  done
}

# How many slots are currently taken. For status output only.
eval_slot_used() {
  local i n=0
  for (( i = 0; i < CF393_EVAL_SLOTS; i++ )); do
    [ -f "$CF393_EVAL_SLOTDIR/slot_$i" ] || continue
    flock -n "$CF393_EVAL_SLOTDIR/slot_$i" true 2>/dev/null || n=$(( n + 1 ))
  done
  printf '%d/%d\n' "$n" "$CF393_EVAL_SLOTS"
}
