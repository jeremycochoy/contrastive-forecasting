#!/bin/bash
# #393 — make a queued cell wait for a free GPU instead of dying at step 0.
#
# Source this and call `gpu_gate <index>` before any CUDA process. The call
# returns once the caller may use that GPU, and holds the claim until the
# calling shell exits.
#
# Why it exists: on 2026-08-05 at 00:49 `arm5_combab_alignT` was started on
# vast A while `arm5_combab_alignS` already held the GPU. A vast.ai box comes
# up in `Exclusive_Process` compute mode and the container cannot change it
# (`nvidia-smi -c 0` -> "Insufficient Permissions"), so the second CUDA
# context died inside `.to(device)` with "CUDA-capable device(s) is/are busy
# or unavailable". The cell was dead one second after launch. It cost no GPU
# time, which is exactly the problem: nothing in the logs looked urgent and
# the cell sat dead for three hours.
#
# Two mechanisms, because neither alone is enough:
#
#   flock   serialises the cells *we* launch. It is the real ordering
#           primitive: an exclusive lock on a per-GPU file, held on fd 9 for
#           the life of the calling shell, so one leg or one eval owns the
#           device end to end and the next waiter is released the moment it
#           exits (including on a kill -9, which a sentinel file would not
#           survive).
#
#   drain   waits for compute apps we did not launch. Another agent session
#           shares these GPUs, and a leftover process from an earlier attempt
#           holds the device just as hard as a cell of ours. flock cannot see
#           either one, so after taking the lock we still wait for the device
#           to report no resident compute apps.
#
# On a `Default`-mode GPU the gate returns immediately. That is deliberate,
# not a fallback: elisa runs two cells per 4090 on purpose, and a gate that
# serialised there would halve the box's throughput.
#
# The gate is advisory for anything that does not call it. It removes the
# race between the cells this experiment launches, which is the one that
# killed a cell.

# Seconds between polls, and the ceiling on a single wait. The ceiling is
# generous on purpose: a cell to bb100k behind another cell is a ~10 h wait,
# and timing out into a crash would recreate the failure being fixed here.
GPU_GATE_POLL="${GPU_GATE_POLL:-30}"
GPU_GATE_TIMEOUT="${GPU_GATE_TIMEOUT:-86400}"
GPU_GATE_LOCKDIR="${GPU_GATE_LOCKDIR:-/tmp}"

gpu_gate_log() { echo "[$(date '+%m-%d %H:%M:%S')] [gpu-gate] $*"; }

# Compute mode of one GPU: Default, Exclusive_Process, Prohibited, ...
# An unreadable nvidia-smi answers "Default" so a box without the tool runs
# rather than blocks.
gpu_gate_mode() {
  nvidia-smi --id="$1" --query-gpu=compute_mode --format=csv,noheader 2>/dev/null \
    | tr -d '[:space:]' || true
}

# PIDs with a CUDA context on one GPU, ours included.
gpu_gate_apps() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d '[:space:]' || true
}

# Block until <index> is usable. Returns 0 when the caller may proceed, 1 if
# it waited past GPU_GATE_TIMEOUT.
gpu_gate() {
  local gpu="${1:-0}" mode lock waited=0 apps

  mode="$(gpu_gate_mode "$gpu")"
  case "$mode" in
    Exclusive_Process|Exclusive_Thread) ;;
    *) return 0 ;;   # Default / unknown: sharing is intended, do not gate
  esac

  lock="$GPU_GATE_LOCKDIR/cf393_gpu${gpu}.lock"
  : >>"$lock" 2>/dev/null || true

  # fd 9 stays open for the life of this shell, so the lock is released when
  # the leg or the eval exits, however it exits.
  exec 9>>"$lock" || { gpu_gate_log "cannot open $lock; proceeding ungated"; return 0; }
  if ! flock -n 9; then
    gpu_gate_log "GPU $gpu ($mode) held by another cell — waiting for the lock"
    if ! flock -w "$GPU_GATE_TIMEOUT" 9; then
      gpu_gate_log "TIMEOUT after ${GPU_GATE_TIMEOUT}s waiting for the lock on GPU $gpu"
      return 1
    fi
  fi

  # Lock taken. Anything still resident is a process outside this experiment,
  # or one of ours that has not finished releasing the device.
  while :; do
    apps="$(gpu_gate_apps "$gpu")"
    [ -z "$apps" ] && break
    if [ "$waited" -ge "$GPU_GATE_TIMEOUT" ]; then
      gpu_gate_log "TIMEOUT after ${waited}s; GPU $gpu still busy (pids: $(echo "$apps" | tr '\n' ' '))"
      return 1
    fi
    [ $(( waited % 300 )) -eq 0 ] && \
      gpu_gate_log "GPU $gpu draining, ${waited}s so far (pids: $(echo "$apps" | tr '\n' ' '))"
    sleep "$GPU_GATE_POLL"
    waited=$(( waited + GPU_GATE_POLL ))
  done

  [ "$waited" -gt 0 ] && gpu_gate_log "GPU $gpu free after ${waited}s"
  return 0
}
