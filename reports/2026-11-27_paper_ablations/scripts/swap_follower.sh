#!/bin/bash
# Move the head follower of one box of #482 to a new score code folder (round
# 3: follow_multi.sh tries a failed head or score again). follower_box.sh is
# the box side: it goes to the box on the input of ssh.
#   MODE=check  the state of the box: its score code, the followers, the
#               processes of a head or a score, the token of each head
#               trainer (the card token or not), the stops with no score,
#               and the last line of the follower log. It changes nothing.
#   MODE=swap   (the default)
#     1. Read the stops of STOPS with no score file on the box. A failed
#        query, or an answer without the state of each stop, gives exit
#        code 2. With no such stop, the swap ends with exit code 0. In the
#        two cases, the swap changes nothing.
#     2. Copy the folder $CF/FOLDER of elisa to $H/ff/FOLDER on the box, with
#        an md5 check. The running follower reads $H/ff/code: the copy writes
#        nothing there.
#     3. Stop each follower of the box, if no head trains and no score runs.
#        The box freezes the followers before it looks, so no head can start
#        between the check and the stop. Exit code 3: a head or a score runs.
#        Exit code 4: no follower PID, but a follower log grew in the last
#        10 min. In the two cases, the followers go on: try again later.
#     4. Move $H/ff/code to $H/ff/code_<time>, and $H/ff/FOLDER to $H/ff/code.
#     5. Start the follower with the start script of the box, unchanged, for
#        the stops of step 1. The follower adds its lines to the same log,
#        follow_<short>_box.log. Exit code 2 if no follower runs 5 s later.
# A swap of a box that has the new code already skips the copy and the move.
# So after a failure, run the swap again.
#   BOX=root@<ip> BOX_PORT=<port> RUN=<run> SHORT=<short> FOLDER=ff_code_<sha> [MODE=check] \
#     bash swap_follower.sh
set -uo pipefail
BOX="${BOX:?}"; PORT="${BOX_PORT:?}"; RUN="${RUN:?}"; SHORT="${SHORT:?}"; FOLDER="${FOLDER:?}"
MODE="${MODE:-swap}"
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
STOPS="${STOPS:-10 25 50 75 100 125 150 166}"
TOKEN="${TOKEN:-$CF/secrets/hf_token_agent.txt}"
H="${BOX_H:-/workspace/hp64}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
LOG="follow_${SHORT}_box.log"
log() { echo "$(date -u +%FT%TZ) $SHORT: $*"; }
fail() { log "ABORT: $1"; exit "${2:-2}"; }
x() { timeout 120 ssh -n -p "$PORT" $OPTS "$BOX" "$@"; }
on_box() { timeout 120 ssh -p "$PORT" $OPTS "$BOX" "bash -s -- $1 $H ${*:2}" < "$HERE/follower_box.sh"; }
sums() { find . -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum | cut -c1-32; }
box_code() { x "cat $H/ff/code/DEPLOYED_COMMIT 2>/dev/null"; }
new_code() { cat "$CF/$FOLDER/DEPLOYED_COMMIT"; }

copy() {  # $CF/FOLDER to $H/ff/FOLDER on the box, then compare their md5
  local a b
  timeout 3600 rsync -a --exclude '__pycache__' -e "ssh -p $PORT $OPTS" "$CF/$FOLDER/" "$BOX:$H/ff/$FOLDER/" < /dev/null \
    || fail "rsync of $FOLDER failed"
  a=$(cd "$CF/$FOLDER" && sums)
  b=$(x "cd $H/ff/$FOLDER && find . -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum" | cut -c1-32)
  [ -n "$b" ] && [ "$a" = "$b" ] || fail "$H/ff/$FOLDER md5 MISMATCH $a $b"
  log "$H/ff/$FOLDER md5 match $a"
}

token_of() {  # "empty" or the MD5 of a token -> the name of the token
  local card
  card=$(printf '%s\n' "$(cat "$TOKEN")" | md5sum | cut -c1-32)
  case "$1" in empty) echo "no token" ;; "$card") echo "the card token" ;; *) echo "another token" ;; esac
}

# The stops of STOPS that the answer of the box gives as open, in OPEN. The
# answer must give "<k> open" or "<k> scored" for each stop of STOPS, in order.
read_stops() {  # <answer>
  local lines i want=($STOPS)
  mapfile -t lines <<< "$1"
  [ "${#lines[@]}" = "${#want[@]}" ] || return 1
  OPEN=""
  for (( i = 0; i < ${#want[@]}; i++ )); do
    case "${lines[i]}" in
      "${want[i]} open") OPEN+="${OPEN:+ }${want[i]}" ;;
      "${want[i]} scored") ;;
      *) return 1 ;;
    esac
  done
}

open_stops() {  # OPEN, else the end of the swap before any change
  local out
  out=$(on_box stops "$RUN" $STOPS) || fail "the query of the stops with no score failed. Nothing is stopped."
  read_stops "$out" || fail "the box gave no state for each stop ('$(echo "$out" | paste -sd';')'). Nothing is stopped."
}

check() {
  local pid sum
  log "score code on the box: $(box_code). $FOLDER: $(new_code)"
  log "followers (PID and folder): $(on_box followers | paste -sd';')"
  log "head or score processes: $(on_box busy | cut -c1-150 | paste -sd';')"
  on_box token | while read -r pid sum; do log "head trainer $pid: $(token_of "$sum")"; done
  if read_stops "$(on_box stops "$RUN" $STOPS)"; then log "stops with no score: $OPEN"
  else log "stops with no score: no valid answer from the box"; fi
  log "follower log: $(x "tail -n 1 $H/ff/results/$LOG")"
}

stop_followers() {
  local out rc
  out=$(on_box stop); rc=$?
  log "stop: $(echo "$out" | cut -c1-150 | paste -sd';')"
  case "$rc" in
    0) ;;
    3) fail "a head or a score runs. The followers go on, and nothing is stopped: try again later." 3 ;;
    4) fail "no follower PID, but a follower log grew in the last 10 min. Nothing is stopped: look at the box (MODE=check)." 4 ;;
    *) fail "the stop of the followers failed (exit code $rc). Look at the box (MODE=check)." ;;
  esac
}

move_code() {
  local out
  out=$(on_box move "$FOLDER") || fail "$out"
  log "$out"
}

start_follower() {  # for the stops of OPEN
  local pids
  x "cd $H/ff && FF_TRAINER=$H/ff/train_heads_tf32.py E_RUNS=$RUN E_STOPS='$OPEN' E_LOG=$LOG bash start_follow_w320e6f6_box.sh" \
    || fail "the start script failed. Run the swap again."
  sleep 5
  pids=$(on_box followers | paste -sd';')
  [ -n "$pids" ] || fail "no follower runs 5 s after the start. Follower log: $(x "tail -n 1 $H/ff/results/$LOG")"
  log "new follower (PID and folder): $pids"
  log "follower log: $(x "tail -n 1 $H/ff/results/$LOG")"
}

swap() {
  open_stops
  [ -n "$OPEN" ] || { log "each stop of STOPS has its score: no follower to start. Nothing is stopped."; return 0; }
  log "stops with no score: $OPEN"
  if [ "$(box_code)" = "$(new_code)" ]; then log "the box has the score code of $FOLDER already"; else copy; fi
  stop_followers
  [ "$(box_code)" = "$(new_code)" ] || move_code
  [ "$(box_code)" = "$(new_code)" ] || fail "the score code of the box is not $FOLDER. Run the swap again."
  start_follower
}

[[ "$FOLDER" =~ ^ff_code_[0-9a-f]{8}$ ]] && [ -s "$CF/$FOLDER/DEPLOYED_COMMIT" ] \
  || fail "FOLDER=$FOLDER: give a folder ff_code_<sha> of $CF with its DEPLOYED_COMMIT"
[ -s "$TOKEN" ] || fail "no token in $TOKEN"
case "$MODE" in check) check ;; swap) swap ;; *) fail "MODE=$MODE: use check or swap" ;; esac
