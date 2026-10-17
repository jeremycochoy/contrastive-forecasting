#!/bin/bash
# freq_family — follow the backbone run `abc_gift`: one wave (run_wave.sh)
# for each stop of a list, when the checkpoint of the stop exists.
#
# For each stop, in the order of the list:
#   1. Wait for the checkpoint `<ckpt folder>/abc_gift*_<stop>k.pth`. It must
#      load, with its optimizer file. The trainer writes the two files in
#      turn and not in one step, so a file that does not load is not
#      complete. After a new start, the trainer names its files
#      `abc_gift_r2_...`: the script takes the file of the last start.
#      When that file does not load, its trainer can write it now, or one
#      load can fail: the script looks again, and does not take the file of
#      an older start. A file that did not load at each look of FF_SETTLE
#      seconds is cut: the script then names it and takes the start before
#      it. The wait ends after FF_WAIT_MAX seconds: the script then stops,
#      because a later stop cannot come before this one.
#   2. Copy the checkpoint to <base>/ckpt/abc_gift/abc_gift_<stop>k.pth. The
#      wave reads the copy, so it does not read the folder of the run again.
#      results/checkpoints.tsv keeps the source and the SHA-256 of each copy.
#   3. Train the heads of the wave (run_wave.sh with FF_SCORE=0).
#   4. Start the scores of the wave in the background (run_wave.sh with
#      FF_TRAIN=0), and go to the next stop. So the scores of a stop run
#      while the heads of the next stop train.
# A wave that fails does not stop the next stops. The script ends with code
# 1 when a wait ended, or when a wave or its scores failed.
#
# A second start skips the work that is done: a stop with a copy does not
# wait, and run_wave.sh skips each head and each score that exists. A second
# follower with other arms can run at the same time: run_wave.sh trains one
# wave at a time.
#
# A deployed follower starts its waves with the code of its own code folder.
# Give each start the score device of the first start (FF_EVAL_DEVICE):
# run_wave.sh refuses a score on another device.
#
# The script reads the folder of `abc_gift` and writes nothing there.
#
# Usage, on elisa, from the code folder:
#   nohup setsid bash follow_abc_gift.sh [arm ...] \
#     >>/home/jupyter/cf_runs/freq_family/results/follow.log 2>&1 </dev/null &
#   FF_STOPS="40 100"     other stops, in thousands of steps
#   FF_WAIT_MAX=43200     the longest wait for one checkpoint, in seconds
#   FF_SETTLE=300         the seconds of looks after which a file that
#                         does not load is cut
# With no arm, each wave holds the control and the 4 family arms.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMS=("$@")
BASE="${FF_BASE:-/home/jupyter/cf_runs/freq_family}"
RUN="${FF_RUN:-abc_gift}"
CKPT_DIR="${FF_ABC_CKPT:-/home/jupyter/cf_runs/abc_gift/ckpt}"
STOPS="${FF_STOPS:-40 100 140 180 200 240 300 360 400 460}"
WAIT_MAX="${FF_WAIT_MAX:-43200}"
SETTLE="${FF_SETTLE:-300}"
POLL="${FF_POLL:-60}"
WAVE="${FF_WAVE:-$HERE/run_wave.sh}"
RES="$BASE/results"
COPIES="$BASE/ckpt/$RUN"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [freq_family follow $RUN] $*"; }

# A checkpoint and its optimizer file both load.
loads(){  # <checkpoint>
  python3 - "$1" <<'PY' >/dev/null 2>&1
import sys, torch
path = sys.argv[1]
torch.load(path, map_location="cpu", weights_only=False)
torch.load(path[:-4] + "_optimizer.pth", map_location="cpu", weights_only=False)
PY
}

# The time of the first look at which a file did not load, for each file.
declare -A BAD_SINCE

# A file that did not load at each look of SETTLE seconds is cut. Before
# that, its trainer can write it now, or one load can fail.
is_cut(){  # <file>
  local since="${BAD_SINCE["$1"]:-}"
  [ "$since" != cut ] || return 0
  if [ -z "$since" ]; then since=$SECONDS; BAD_SINCE["$1"]=$since; fi
  (( SECONDS - since >= SETTLE )) || return 1
  log "cut: $1 did not load for $SETTLE s. The start before it is next."
  BAD_SINCE["$1"]=cut
}

# The checkpoint of a stop, in CKPT: the file of the last start, when it
# loads. A file of a later start that does not load ends the search until it
# is cut: the script does not take the file of an older start while a
# trainer writes this one.
find_ckpt(){  # <stop k>
  local f
  while read -r f; do
    [ -n "$f" ] || continue
    if [ -s "${f%.pth}_optimizer.pth" ] && loads "$f"; then CKPT="$f"; return 0; fi
    is_cut "$f" || return 1
  done < <(ls "$CKPT_DIR/$RUN"*_"$1"k.pth 2>/dev/null | sort -rV)
  return 1
}

# Wait for the checkpoint of a stop. Returns 1 when the wait ends first.
wait_ckpt(){  # <stop k>
  local start=$SECONDS said=-1
  until find_ckpt "$1"; do
    if (( SECONDS - start >= WAIT_MAX )); then return 1; fi
    if (( (SECONDS - start) / 1800 > said )); then
      said=$(( (SECONDS - start) / 1800 ))
      log "waiting for the checkpoint of ${1}k in $CKPT_DIR ($(( SECONDS - start )) s of $WAIT_MAX)"
    fi
    sleep "$POLL"
  done
}

# The copy of the checkpoint of a stop that the wave reads, in COPY. The
# copy goes through a file of this process: a second follower can copy the
# same checkpoint at the same time.
copy_ckpt(){  # <stop k>
  local bytes sum tmp
  COPY="$COPIES/${RUN}_${1}k.pth"
  [ -s "$COPY" ] && return 0
  wait_ckpt "$1" || return 1
  bytes=$(stat -c %s "$CKPT")
  tmp="$COPY.tmp.$$"
  cp "$CKPT" "$tmp" && [ "$(stat -c %s "$tmp")" = "$bytes" ] \
    && mv -f "$tmp" "$COPY" \
    || { rm -f "$tmp"; log "ABORT: the copy of $CKPT is not $bytes bytes"; return 2; }
  sum=$(sha256sum "$COPY" | cut -d' ' -f1)
  [ -s "$RES/checkpoints.tsv" ] \
    || printf 'run\tstop_k\tsource\tbytes\tsha256\n' >"$RES/checkpoints.tsv"
  printf '%s\t%s\t%s\t%s\t%s\n' "$RUN" "$1" "$CKPT" "$bytes" "$sum" >>"$RES/checkpoints.tsv"
  log "checkpoint ${1}k: $CKPT ($bytes bytes, sha256 $sum)"
}

[ -f "$WAVE" ] || { log "ABORT: no wave script at $WAVE"; exit 2; }
mkdir -p "$RES" "$COPIES" || exit 2
log "start: stops $STOPS, arms ${ARMS[*]:-control and the 4 family arms}"
rc=0; pids=(); names=()
for stop in $STOPS; do
  copy_ckpt "$stop"
  case $? in
    0) ;;
    1) log "STOP: no checkpoint of ${stop}k after $WAIT_MAX s. The later stops get no wave."
       rc=1; break ;;
    *) rc=1; continue ;;
  esac
  if ! FF_SCORE=0 bash "$WAVE" "$RUN" "$stop" "$COPY" "${ARMS[@]}"; then
    log "the heads of ${stop}k FAILED. The next stop goes on."
    rc=1; continue
  fi
  FF_TRAIN=0 bash "$WAVE" "$RUN" "$stop" "$COPY" "${ARMS[@]}" &
  pids+=($!); names+=("$stop")
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { log "the scores of ${names[$i]}k FAILED"; rc=1; }
done
log "FOLLOW_END rc=$rc"
exit "$rc"
