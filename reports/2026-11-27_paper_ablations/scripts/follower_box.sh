#!/bin/bash
# The box side of swap_follower.sh (#482, round 3). swap_follower.sh sends this
# file on the input of ssh:
#   ssh <box> "bash -s -- <command> <box folder H> [<arg> ...]" < follower_box.sh
# The processes of the head follower of the box are the processes with
# FF_BASE=$H/ff in their environment: each follower (bash follow_multi.sh, and
# its forks) and each process that a follower starts. The trainer of the run
# has no FF_BASE.
#   followers             the PID and the code folder of each follower
#   busy                  each process of the follower that is not a follower
#                         and not a sleep: a head trainer, a score, a wait for
#                         the stream lock, a load check of a checkpoint
#   stop                  if nothing is busy, end each follower by its PID
#   move <folder>         $H/ff/code -> $H/ff/code_<time>, then
#                         $H/ff/<folder> -> $H/ff/code
#   stops <run> <k> ...   the stops of the list with no score file
#   token                 for each head trainer: its PID, and "empty" or the
#                         MD5 of its HF_TOKEN (never the token)
set -uo pipefail
CMD="${1:?command}"; H="${2:?box folder}"; shift 2
BASE="$H/ff"

procs(){  # "<pid> <command line>" of each process with FF_BASE=$BASE
  local d
  for d in /proc/[0-9]*; do
    grep -qzxF -- "FF_BASE=$BASE" "$d/environ" 2>/dev/null || continue
    echo "${d#/proc/} $(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)"
  done
}
is_follower='$2 ~ /(^|\/)bash$/ && $3 ~ /(^|\/)follow_multi\.sh$/'
follower_pids(){ procs | awk "$is_follower"' { print $1 }'; }
busy(){ procs | awk "!($is_follower) && \$2 !~ /(^|\\/)sleep\$/"; }
followers(){
  local pid
  for pid in $(follower_pids); do echo "$pid $(readlink "/proc/$pid/cwd")"; done
}

stop(){
  local pids i
  [ -z "$(busy)" ] || { echo "BUSY:"; busy; return 3; }
  pids=$(follower_pids)
  [ -n "$pids" ] || { echo "no follower"; return 0; }
  kill $pids
  for (( i = 0; i < 30; i++ )); do
    [ -z "$(follower_pids)" ] && { echo "stopped:" $pids; return 0; }
    sleep 1
  done
  echo "ABORT: a follower runs 30 s after its stop:" $(follower_pids); return 1
}

move(){  # <folder>
  local new="$BASE/$1" old="$BASE/code_$(date -u +%m%d_%H%M%S)"
  [ -z "$(follower_pids)" ] || { echo "ABORT: a follower runs"; return 1; }
  [ -d "$new" ] && [ -d "$BASE/code" ] || { echo "ABORT: no $new or no $BASE/code"; return 1; }
  mv "$BASE/code" "$old" || { echo "ABORT: no move of $BASE/code"; return 1; }
  mv "$new" "$BASE/code" || { mv "$old" "$BASE/code"; echo "ABORT: no move of $new"; return 1; }
  echo "moved: $BASE/code -> $old, $new -> $BASE/code"
}

stops(){  # <run> <stop k> ...
  local run="$1" k out=(); shift
  for k in "$@"; do [ -s "$BASE/results/score_${run}_bb${k}k_h30k_control.txt" ] || out+=("$k"); done
  echo "${out[*]}"
}

token(){
  local pid value
  for pid in $(procs | awk '/ --jobs / { print $1 }'); do
    value=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^HF_TOKEN=//p')
    if [ -z "$value" ]; then echo "$pid empty"; else echo "$pid $(printf '%s\n' "$value" | md5sum | cut -c1-32)"; fi
  done
}

case "$CMD" in
  followers|busy|stop|move|stops|token) "$CMD" "$@" ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
