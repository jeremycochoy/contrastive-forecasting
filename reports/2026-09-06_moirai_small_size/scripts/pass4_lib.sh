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
