#!/bin/bash
# #407 — the draws card 1 owes, in the order card 1 runs them.
#
# The watchdog fires the band at the LAST stop, 665,000 steps. Three more
# bands belong to this card, and none of them fits the watchdog's rule:
#
#   200k re-draw  the protocol seed 20260722, drawn again on this machine
#                 and on this code. #373 published 1.0660 from another
#                 round and another box, so that number carried head seed,
#                 machine and code version together. This draw holds the
#                 seed still and moves only the machine and the code.
#                 DONE. Both heads reproduce #373 exactly, delta +0.0000.
#   300k band     seeds 20260723 and 20260724 at the 300,000-step stop.
#                 Card 1 stands idle between the 300k checkpoint and the
#                 450k one, about 15 hours. This band costs that idle time
#                 and nothing else, and it turns 300k from one draw per
#                 head into a band. The card then reads a band at 200k,
#                 300k, 450k and 665k instead of two stops only.
#   450k band     the same two seeds at 450,000 steps. The interior stops
#                 carry one draw each, so a move between 300k and 450k has
#                 no scale beside it.
#
# Every stage waits on card 1 rather than on a clock. A `ckpt` stage also
# waits for its backbone to land. `head_vram_gate` serialises the GPU on
# one flock, so this queue runs ONE band at a time and a second band only
# queues behind the first.
#
# This takes no GPU time from the driver: `BAND_GPU` is the card the driver
# is not on, and the tag of every draw carries its seed, so no draw can
# overwrite the card's own six numbers.
#
# Usage:
#   WT=<checkout> RUNS=<durable root> BAND_GPU=1 \
#     nohup setsid bash band_queue.sh > band_queue.out 2>&1 &
#
# The queue reads its state off the disk at every start, so a restart never
# repeats a band that already scored and never loses one that did not.
#
# QUEUE_PERIOD     seconds between checks (default 300).
# QUEUE_MAX_FIRES  give up on a stage after this many launches (default 4).
#                  A draw that dies at once would otherwise re-launch every
#                  period for the whole 40 hours of the card.
# QUEUE_ONCE       run one pass of the loop and exit. The test seam.
# QUEUE_DRY        decide, log, launch nothing. The other test seam.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="${CF407_RESULTS:-$STUDY/results}"
mkdir -p "$RES"

export WT="${WT:-/tmp/contrastive-forecasting-407}"
export RUNS="${RUNS:-/home/jupyter/cf373_r3/sync}"
export BAND_GPU="${BAND_GPU:-1}"
PARENT_RES="$WT/reports/2026-08-08_rollout_depth/results"

PERIOD="${QUEUE_PERIOD:-300}"
MAX_FIRES="${QUEUE_MAX_FIRES:-4}"
ONCE="${QUEUE_ONCE:-0}"
DRY="${QUEUE_DRY:-0}"
LOG="$RES/band_queue.log"

# One line per stage: <stop steps>|<head seeds>|<gate>.
#   now   fire as soon as card 1 is free.
#   ckpt  fire when that stop's backbone is on disk. It fires on the
#         CHECKPOINT and not on the score, so the two extra seeds train
#         while the driver still scores the protocol seed at the same stop.
# The order of the lines is the order card 1 runs them.
STAGES=(
  "200000|20260722|now"
  "300000|20260723 20260724|ckpt"
  "450000|20260723 20260724|ckpt"
)
# The test seam takes one stage per LINE, so a seed list may hold spaces.
[ -n "${QUEUE_STAGES:-}" ] && mapfile -t STAGES <<<"$QUEUE_STAGES"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [cf407-queue] $*" | tee -a "$LOG"; }

SCRIPT="$(readlink -f "$HERE/replicate_heads.sh" 2>/dev/null)"

# Every chain of THIS checkout's band script, as "<pid> <stop>" lines.
#
# `pgrep -f replicate_heads.sh` alone also matches the shell that launched
# one and any tail that watches it, so this reads `argv[1]` and `argv[2]`
# out of `/proc`. It then resolves `argv[1]` against the process's own
# working directory and demands this checkout's copy. The bands launch by
# a RELATIVE path, so a basename test is not enough: a second worktree of
# this repo would read as "a band is up" and this queue would wait forever.
live_chains(){
  local p a1 a2 cwd full
  for p in $(pgrep -f 'replicate_heads\.sh' 2>/dev/null); do
    a1=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 2p)
    a2=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n 3p)
    case "$a1" in */replicate_heads.sh) ;; *) continue;; esac
    case "$a1" in
      /*) full="$a1" ;;
      *)  cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null) || continue
          [ -n "$cwd" ] || continue
          full="$cwd/$a1" ;;
    esac
    full=$(readlink -f "$full" 2>/dev/null)
    [ -n "$full" ] && [ "$full" = "$SCRIPT" ] && echo "$p $a2"
  done
}

# A band for one stop is up.
replicate_alive(){ # <stop>
  live_chains | awk -v s="$1" '$2 == s { found = 1 } END { exit !found }'
}

# Card 1 carries a band for ANY stop. One band at a time keeps this study
# at two concurrent GIFT-Evals, which is 8 of elisa's 32 cores.
any_replicate_alive(){ [ -n "$(live_chains)" ]; }

# Both heads of one (stop, seed) are scored.
seed_drained(){ # <stop> <seed>
  local k=$(( $1 / 1000 )) head
  for head in student teacher; do
    [ -s "$PARENT_RES/score_A4_k3_bb${k}k_${head}_s${2}.txt" ] || return 1
  done
  return 0
}

# Every draw of one stage is scored.
stage_drained(){ # <stop> <seed...>
  local stop="$1" seed; shift
  for seed in "$@"; do
    seed_drained "$stop" "$seed" || return 1
  done
  return 0
}

# The scores of one stage, on one line.
stage_scores(){ # <stop> <seed...>
  local k=$(( $1 / 1000 )) seed head out=""; shift
  for seed in "$@"; do
    for head in student teacher; do
      out="$out  s${seed} ${head} $(cat \
        "$PARENT_RES/score_A4_k3_bb${k}k_${head}_s${seed}.txt" 2>/dev/null)"
    done
  done
  echo "$out"
}

# That stop's backbone is on disk, whole.
#
# The gate also demands the `_optimizer.pth` sidecar. `save_snapshot` in
# `train.py` writes the backbone FIRST and the sidecar second, so a sidecar
# on disk proves the backbone write finished. Without that test this queue
# could glob a checkpoint the driver is still writing, and the band would
# then load a truncated file and die.
ckpt_here(){ # <stop>
  local f
  f="$(python3 "$HERE/full_pass.py" --ckpt-at "$1" --root "$RUNS" 2>/dev/null)"
  [ -n "$f" ] && [ -s "$f" ] && [ -s "${f%.pth}_optimizer.pth" ]
}

fire(){ # <stop> <seed...>
  local stop="$1"; shift
  local k=$(( stop / 1000 ))
  log "FIRE band at ${k}k, seeds $*, on GPU $BAND_GPU"
  [ "$DRY" = 1 ] && { log "QUEUE_DRY — no launch"; return 0; }
  WT="$WT" CF373_ROOT="$RUNS" BB_GPU="$BAND_GPU" \
    HEAD_VRAM_MIB="${BAND_VRAM_MIB:-6400}" \
    HEAD_VRAM_TIMEOUT="${BAND_VRAM_TIMEOUT:-28800}" \
    nohup setsid bash "$HERE/replicate_heads.sh" "$stop" "$@" \
      >>"$RES/replicate_${k}k.out" 2>&1 &
}

# Read the state of every stage off the disk.
STOP=(); SEEDS=(); GATE=(); STATE=(); FIRES=()
for row in "${STAGES[@]}"; do
  stop="${row%%|*}"; rest="${row#*|}"
  seeds="${rest%%|*}"; gate="${rest##*|}"
  STOP+=("$stop"); SEEDS+=("$seeds"); GATE+=("$gate"); FIRES+=(0)
  # shellcheck disable=SC2086
  if stage_drained "$stop" $seeds; then
    STATE+=(done)
  elif [ -s "$RES/replicate_$(( stop / 1000 ))k.log" ]; then
    STATE+=(fired)
  else
    STATE+=(pending)
  fi
done

log "start period=${PERIOD}s gpu=$BAND_GPU stages=${#STOP[@]}"
for i in "${!STOP[@]}"; do
  log "  stage $(( i + 1 )): $(( STOP[i] / 1000 ))k seeds ${SEEDS[$i]} gate ${GATE[$i]} state ${STATE[$i]}"
done

while :; do
  for i in "${!STOP[@]}"; do
    st="${STATE[$i]}"
    case "$st" in done|lost) continue;; esac
    stop="${STOP[$i]}"; seeds="${SEEDS[$i]}"; k=$(( stop / 1000 ))

    # shellcheck disable=SC2086
    if stage_drained "$stop" $seeds; then
      # shellcheck disable=SC2086
      log "band at ${k}k DONE:$(stage_scores "$stop" $seeds)"
      STATE[$i]=done; continue
    fi
    # Its own chains run. Nothing to decide.
    replicate_alive "$stop" && continue

    if [ "$st" = fired ]; then
      log "WARN: the ${k}k band is gone and not scored. It goes back in the queue."
      STATE[$i]=pending
    fi
    # Card 1 takes one band at a time.
    any_replicate_alive && continue
    [ "${GATE[$i]}" = ckpt ] && ! ckpt_here "$stop" && continue
    if [ "${FIRES[$i]}" -ge "$MAX_FIRES" ]; then
      log "GIVING UP on the ${k}k band: ${FIRES[$i]} launches and no score"
      STATE[$i]=lost; continue
    fi
    FIRES[$i]=$(( FIRES[$i] + 1 ))
    # shellcheck disable=SC2086
    fire "$stop" $seeds
    STATE[$i]=fired
    break
  done

  left=0
  for st in "${STATE[@]}"; do
    case "$st" in done|lost) ;; *) left=$(( left + 1 ));; esac
  done
  [ "$left" -eq 0 ] && { log "nothing left to fire — the queue stops"; exit 0; }

  [ "$ONCE" = 1 ] && { log "QUEUE_ONCE — one pass done"; exit 0; }
  sleep "$PERIOD"
done
