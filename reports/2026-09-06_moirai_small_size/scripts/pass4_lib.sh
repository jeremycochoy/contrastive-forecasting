#!/bin/bash
# #412 pass 4 — the two process lookups the status block and the await share.
#
# A grep over `ps` args is NOT enough for this pass. `k3_r100_09_lr56_dec` is a
# PREFIX of `k3_r100_09_lr56_dec10k`, so a name match reads one arm as the
# other, and the two arms are the whole pass. Both lookups below match on a
# value that is unique to one leg.
#
# Source it after `study.sh`.

# The trainer of ONE leg, by its `--save-dir`. That path holds the arm and the
# stop, and no two legs share one.
cf412_trainer_pid(){  # <arm> <stop>
  local dir pid
  dir="$(cf412_leg_dir "$1" "$2")"
  for pid in $(pgrep -f 'train\.py' 2>/dev/null); do
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null \
      | grep -qF -- "--save-dir $dir " && { echo "$pid"; return 0; }
  done
  return 1
}

# The head driver of ONE leg. `head_eval.sh <arm> <stop>` carries both, so the
# match takes the two arguments together and ends at the stop.
cf412_head_pid(){  # <arm> <stop>
  local pid
  for pid in $(pgrep -f 'head_eval' 2>/dev/null); do
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null \
      | grep -qE "head_eval(_bb)?\.sh $1 $2( |$)" && { echo "$pid"; return 0; }
  done
  return 1
}

# The count of processes that match a pattern, WITHOUT the agent shells.
#
# An agent session runs each command through a wrapper whose own arguments hold
# the whole command text, so `pgrep -f head_eval` matches the shell that asks
# the question. Those wrappers all carry `shell-snapshots`, and no lane does.
cf412_count_real(){  # <extended regex>
  local pid n=0
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    grep -qa 'shell-snapshots' "/proc/$pid/cmdline" 2>/dev/null && continue
    n=$(( n + 1 ))
  done
  echo "$n"
}

# The count of lane drivers that can still start a leg of this card. Pass 4
# runs under `phase1.sh`, which trains the head between two legs of one arm.
# Earlier passes ran `pass2_lane.sh`, and a mixed box holds both.
cf412_lanes_running(){
  cf412_count_real '(phase1|pass2_lane|pass5_lane)\.sh'
}

# The count of head drivers on the box, of any arm.
cf412_heads_running(){
  cf412_count_real 'head_eval(_bb)?\.sh '
}

# The card a head driver runs on, or nothing. `head_eval.sh` takes it from the
# environment, so the process holds it.
cf412_head_gpu(){  # <pid>
  tr '\0' '\n' <"/proc/${1:?pid}/environ" 2>/dev/null \
    | sed -n 's/^BB_GPU=//p' | head -1
}

# The free memory of one card, in MiB, or nothing.
cf412_gpu_free(){  # <gpu index>
  nvidia-smi --id="${1:?gpu}" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

# Every descendant of one pid, that pid first.
cf412_descendants(){  # <pid>
  local k
  echo "${1:?pid}"
  for k in $(pgrep -P "$1" 2>/dev/null); do cf412_descendants "$k"; done
}

# A head that is ALIVE but cannot start. `head_eval_bb.sh` takes the card head
# lock FIRST and then waits up to HEAD_VRAM_TIMEOUT (4 hours) for
# CF412_HEAD_VRAM_MIB of free memory. So a blocked head looks exactly like a
# working head in the process table, and it holds that lock the whole time.
#
# THE CARD'S FREE MEMORY ALONE DOES NOT ANSWER THIS. A head that clears the
# gate then allocates about 6,250 MiB itself, which drops the card BACK under
# the gate. Read by free memory alone, every working head reads as blocked.
# So the test is the head's own allocation: a head that holds GPU memory is
# past the gate, whatever the card reports now.
#
# Prints "<gpu> <free>" when the head is alive, holds no GPU memory, and its
# card is short. Prints nothing otherwise.
cf412_head_blocked(){  # <arm> <stop>
  local pid gpu free p gpids need="${CF412_HEAD_VRAM_MIB:-9000}"
  pid="$(cf412_head_pid "$1" "$2")" || return 1
  gpids=" $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | tr '\n' ' ')"
  for p in $(cf412_descendants "$pid"); do
    case "$gpids" in *" $p "*) return 1 ;; esac
  done
  gpu="$(cf412_head_gpu "$pid")"; [ -n "$gpu" ] || return 1
  free="$(cf412_gpu_free "$gpu")"; [ -n "$free" ] || return 1
  [ "$free" -lt "$need" ] || return 1
  echo "$gpu $free"
}
