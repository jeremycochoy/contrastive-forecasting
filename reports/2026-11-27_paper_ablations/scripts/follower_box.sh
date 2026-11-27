#!/bin/bash
# The box side of swap_follower.sh (#482, rounds 3 and 4). swap_follower.sh
# sends this file on the input of ssh:
#   ssh <box> "bash -s -- <command> <box folder H> [<arg> ...]" < follower_box.sh
# The processes of the head follower of the box are the processes with
# FF_BASE=$H/ff in their environment: each follower (bash follow_multi.sh, and
# its forks) and each process that a follower starts. The trainer of the run
# has no FF_BASE.
#   followers             the PID and the code folder of each follower
#   busy                  each process of the follower that is not a follower
#                         and not a sleep: a head trainer, a score, a wait for
#                         the stream lock, a load check of a checkpoint
#   stop                  freeze each follower (SIGSTOP): a frozen follower
#                         cannot start a head or a score. Then look for busy
#                         processes on the box and in the children of the
#                         frozen followers. If one runs, resume the followers
#                         (SIGCONT) and exit 3. Else kill the frozen followers
#                         (SIGKILL). With no follower PID, exit 4 if a
#                         follower log of $H/ff/results grew in the last 10 min.
#   move <folder>         if no follower and no busy process runs:
#                         $H/ff/code -> $H/ff/code_<time>, then
#                         $H/ff/<folder> -> $H/ff/code
#   stops <run> <k> ...   "<k> open" or "<k> scored" for each stop of the list
#   token                 for each head trainer: its PID, and "empty" or the
#                         MD5 of its HF_TOKEN (never the token)
set -uo pipefail
# The scans need "mapfile -d" (bash 4.4): with an older bash, they find no process.
(( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] >= 404 )) || { echo "ABORT: bash $BASH_VERSION, not 4.4 or later"; exit 2; }
CMD="${1:?command}"; H="${2:?box folder}"; shift 2
BASE="$H/ff"
FROZEN=()   # the PIDs of the followers that stop froze
KILLED=0    # 1 after stop killed them

# The scans read /proc with no fork for each process: the followers stay
# frozen for a shorter time.
line(){  # <process folder>: "<pid> <command line>"
  local cmd
  { mapfile -t -d '' cmd < "$1/cmdline"; } 2>/dev/null
  echo "${1#/proc/} ${cmd[*]}"
}
procs(){  # each process with FF_BASE=$BASE
  local d all one
  for d in /proc/[0-9]*; do
    { mapfile -t -d '' all < "$d/environ"; } 2>/dev/null || continue
    printf -v one '\n%s' "${all[@]}"
    [[ "$one"$'\n' == *$'\n'"FF_BASE=$BASE"$'\n'* ]] && line "$d"
  done
}
kids(){  # each child of a frozen follower, not a zombie
  local d s f
  [ "${#FROZEN[@]}" -gt 0 ] || return 0
  for d in /proc/[0-9]*; do
    { read -r s < "$d/stat"; } 2>/dev/null || continue
    f=(${s##*) })   # the state, the parent PID, ...
    [ "${f[0]}" != Z ] && [[ " ${FROZEN[*]} " == *" ${f[1]} "* ]] && line "$d"
  done
}
is_follower='$2 ~ /(^|\/)bash$/ && $3 ~ /(^|\/)follow_multi\.sh$/'
follower_pids(){ procs | awk "$is_follower"' { print $1 }'; }
awake(){ { procs; kids; } | awk '$2 !~ /(^|\/)sleep$/' | sort -un; }   # followers, heads, scores
busy(){ awake | awk "!($is_follower)"; }
followers(){
  local pid
  for pid in $(follower_pids); do echo "$pid $(readlink "/proc/$pid/cwd")"; done
}

freeze(){  # SIGSTOP each follower, until no new follower comes
  local i pid new
  for (( i = 0; i < 10; i++ )); do
    new=()
    for pid in $(follower_pids); do [[ " ${FROZEN[*]} " == *" $pid "* ]] || new+=("$pid"); done
    [ "${#new[@]}" -gt 0 ] || return 0
    kill -STOP "${new[@]}" 2>/dev/null
    FROZEN+=("${new[@]}")
  done
  return 1
}
thaw(){  # SIGCONT each frozen follower that stop did not kill
  [ "$KILLED" = 1 ] || [ "${#FROZEN[@]}" -eq 0 ] || kill -CONT "${FROZEN[@]}" 2>/dev/null
  FROZEN=()
}
gone(){  # no follower and no busy process, within 30 s
  local i
  for (( i = 0; i < 30; i++ )); do
    [ -z "$(awake)" ] && return 0
    sleep 1
  done
  echo "ABORT: a process runs 30 s after the stop:"; awake; return 1
}
no_follower(){  # exit code 4 if a follower log grew in the last 10 min
  local grew
  grew=$(find "$BASE/results" -maxdepth 1 -name 'follow_*.log' -mmin -10 2>/dev/null)
  [ -z "$grew" ] || { echo "NO PID: no follower PID, but a follower log grew in the last 10 min:" $grew; return 4; }
  echo "no follower"
}

stop(){
  local work
  freeze || { thaw; echo "ABORT: new followers come after 10 freezes"; return 1; }
  work=$(busy)
  [ -z "$work" ] || { thaw; echo "BUSY:"; echo "$work"; return 3; }
  [ "${#FROZEN[@]}" -gt 0 ] || { no_follower; return; }
  kill -KILL "${FROZEN[@]}" 2>/dev/null; KILLED=1
  gone && echo "stopped: ${FROZEN[*]}"
}

move(){  # <folder>
  local new="$BASE/$1" old="$BASE/code_$(date -u +%m%d_%H%M%S)"
  [ -z "$(awake)" ] || { echo "ABORT: a follower, a head or a score runs:"; awake; return 1; }
  [ -d "$new" ] && [ -d "$BASE/code" ] || { echo "ABORT: no $new or no $BASE/code"; return 1; }
  mv "$BASE/code" "$old" || { echo "ABORT: no move of $BASE/code"; return 1; }
  mv "$new" "$BASE/code" || { mv "$old" "$BASE/code"; echo "ABORT: no move of $new"; return 1; }
  echo "moved: $BASE/code -> $old, $new -> $BASE/code"
}

stops(){  # <run> <stop k> ...
  local run="$1" k; shift
  [ -d "$BASE/results" ] || { echo "ABORT: no $BASE/results"; return 1; }
  for k in "$@"; do
    if [ -s "$BASE/results/score_${run}_bb${k}k_h30k_control.txt" ]; then echo "$k scored"; else echo "$k open"; fi
  done
}

token(){
  local pid value
  for pid in $(procs | awk '/ --jobs / { print $1 }'); do
    value=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^HF_TOKEN=//p')
    if [ -z "$value" ]; then echo "$pid empty"; else echo "$pid $(printf '%s\n' "$value" | md5sum | cut -c1-32)"; fi
  done
}

# A lost ssh link or a signal must not leave a follower frozen.
trap thaw EXIT
trap 'exit 1' INT TERM
trap '' HUP PIPE
case "$CMD" in
  followers|busy|stop|move|stops|token) "$CMD" "$@" ;;
  *) echo "unknown command $CMD"; exit 2 ;;
esac
