#!/bin/bash
# Move the head follower of one box of #482 to a new score code folder (round
# 3: follow_multi.sh tries a failed head or score again). follower_box.sh is
# the box side: it goes to the box on the input of ssh.
#   MODE=check  the state of the box: its score code, the followers, the
#               processes of a head or a score, the token of each head
#               trainer (the card token or not), the stops with no score,
#               and the last line of the follower log. It changes nothing.
#   MODE=swap   (the default)
#     1. Copy the folder $CF/FOLDER of elisa to $H/ff/FOLDER on the box, with
#        an md5 check. The running follower reads $H/ff/code: the copy writes
#        nothing there.
#     2. Stop each follower of the box by its PID, if no head trains and no
#        score runs. Else end with exit code 3 and stop nothing: try again
#        later.
#     3. Move $H/ff/code to $H/ff/code_<time>, and $H/ff/FOLDER to $H/ff/code.
#     4. Start the follower with the start script of the box, unchanged, for
#        the stops of STOPS with no score file on the box. The follower adds
#        its lines to the same log, follow_<short>_box.log.
# A swap of a box that has the new code already skips steps 1 and 3. So after
# a failure, run the swap again.
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

check() {
  local pid sum
  log "score code on the box: $(box_code). $FOLDER: $(new_code)"
  log "followers (PID and folder): $(on_box followers | paste -sd';')"
  log "head or score processes: $(on_box busy | cut -c1-150 | paste -sd';')"
  on_box token | while read -r pid sum; do log "head trainer $pid: $(token_of "$sum")"; done
  log "stops with no score: $(on_box stops "$RUN" $STOPS)"
  log "follower log: $(x "tail -n 1 $H/ff/results/$LOG")"
}

stop_followers() {
  local out rc
  out=$(on_box stop); rc=$?
  log "stop: $(echo "$out" | cut -c1-150 | paste -sd';')"
  [ "$rc" != 3 ] || fail "a head or a score runs. Nothing is stopped: try again later." 3
  [ "$rc" = 0 ] || fail "the stop of the followers failed"
}

move_code() {
  local out
  out=$(on_box move "$FOLDER") || fail "$out"
  log "$out"
}

start_follower() {
  local stops
  stops=$(on_box stops "$RUN" $STOPS)
  [ -n "$stops" ] || { log "each stop has its score: no follower to start"; return 0; }
  x "cd $H/ff && FF_TRAINER=$H/ff/train_heads_tf32.py E_RUNS=$RUN E_STOPS='$stops' E_LOG=$LOG bash start_follow_w320e6f6_box.sh" \
    || fail "the start script failed. Run the swap again."
  sleep 5
  log "new follower (PID and folder): $(on_box followers | paste -sd';')"
  log "follower log: $(x "tail -n 1 $H/ff/results/$LOG")"
}

swap() {
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
