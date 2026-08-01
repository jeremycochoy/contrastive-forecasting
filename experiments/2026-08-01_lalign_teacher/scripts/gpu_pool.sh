#!/bin/bash
# #390 — a fixed pool of GPU slots, shared by the backbone and eval stages.
#
# `orchestrate.sh` runs the cells as sequential PAIRS, one per GPU. Measured
# on elisa (`results/probe_concurrency.txt`), a second trainer on the SAME
# 4090 costs each one only ~10% of its step rate:
#
#   1 trainer  on GPU 1 : 3.2 sps
#   2 trainers on GPU 1 : 2.9 + 2.8 = 5.7 sps aggregate   (1.8x)
#
# So two slots per GPU nearly halve the wall clock of a multi-day sweep. This
# helper is the piece that makes that possible without touching the reviewed
# pair-at-a-time orchestrator: it keeps `SLOTS_PER_GPU` jobs alive on each of
# the two GPUs and starts the next arm the moment a slot frees, instead of
# joining a whole pair before starting the next one.
#
# Usage:
#   source gpu_pool.sh
#   my_job(){ local arm="$1" gpu="$2"; ...; }     # returns the arm's rc
#   POOL_RC_DIR=/some/scratch pool_run 2 my_job arm5 arm5_tr1 ...
#
# pool_run sets POOL_RC[<arm>] to that arm's own exit status — each job
# records its own rc to a file, because `wait -n` reports the status of
# whichever child happened to exit and cannot be attributed to a PID after
# the fact. pool_run itself always returns 0: one dead cell does not stop the
# sweep (the cells are independent, and a dropped cell is the issue's own
# stop rule).
set -uo pipefail

declare -A POOL_RC=()

pool_run() { # slots_per_gpu job_fn arm...
  local slots="$1" job="$2"; shift 2
  local arms=("$@")
  local n_gpus="${POOL_N_GPUS:-2}"
  local rcdir="${POOL_RC_DIR:-$(mktemp -d)}"
  mkdir -p "$rcdir"
  local -A pid_arm=() pid_gpu=() gpu_used=()
  local g i=0 arm gpu pid
  for (( g = 0; g < n_gpus; g++ )); do gpu_used[$g]=0; done
  POOL_RC=()

  _pool_free_gpu() {
    local g
    for (( g = 0; g < n_gpus; g++ )); do
      if [ "${gpu_used[$g]}" -lt "$slots" ]; then echo "$g"; return 0; fi
    done
    return 1
  }

  _pool_reap() {  # harvest every child that has exited; 0 if any was reaped
    local pid arm gpu got=1
    for pid in "${!pid_arm[@]}"; do
      kill -0 "$pid" 2>/dev/null && continue
      arm="${pid_arm[$pid]}"; gpu="${pid_gpu[$pid]}"
      POOL_RC[$arm]="$(cat "$rcdir/$arm.rc" 2>/dev/null || echo 99)"
      gpu_used[$gpu]=$(( gpu_used[$gpu] - 1 ))
      unset 'pid_arm[$pid]' 'pid_gpu[$pid]'
      got=0
    done
    return $got
  }

  while [ "$i" -lt "${#arms[@]}" ] || [ "${#pid_arm[@]}" -gt 0 ]; do
    while [ "$i" -lt "${#arms[@]}" ] && gpu="$(_pool_free_gpu)"; do
      arm="${arms[$i]}"
      rm -f "$rcdir/$arm.rc"
      ( "$job" "$arm" "$gpu"; echo $? > "$rcdir/$arm.rc" ) & pid=$!
      pid_arm[$pid]="$arm"; pid_gpu[$pid]="$gpu"
      gpu_used[$gpu]=$(( gpu_used[$gpu] + 1 ))
      i=$(( i + 1 ))
      # Stagger the starts: two trainers hitting the HF stream and CUDA init
      # in the same second is the one way this pool differs from launching
      # them by hand, and it is worth the five seconds.
      sleep 5
    done
    [ "${#pid_arm[@]}" -gt 0 ] || break
    wait -n 2>/dev/null   # block until at least one child exits
    _pool_reap || true
  done
  return 0
}
