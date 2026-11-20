#!/bin/bash
# Pauses (or stops) one width-64 run right after a save: stops the trainer <pid>
# right after its save <k>k and the optimizer file of that save load, then writes the file <run dir>/stop, so that
# the supervisor of the run ends after its 120 s sleep. To resume: rm <run dir>/stop, then start the supervisor
# again with BB_GPU (its run script resumes from the last save that loads). Looks every 15 s, at most <max wait> s
# (default 3 h).
#   setsid nohup bash pause_after_save.sh <run dir> <pid> <k> [max wait] [stop text] >> <run dir>/logs/pause.log 2>&1 < /dev/null &
# The stop text goes into <run dir>/stop (default: 'paused after the <k>k save'). In place of <k>, give the full path
# of the save: a resume branch writes <run>_r<n>_<k>k.pth.
set -uo pipefail
DIR="${1:?run dir}"; PID="${2:?trainer pid}"; K="${3:?save step in thousands, or the save path}"; MAX_WAIT="${4:-10800}"
case "$K" in
  *.pth) CKPT="$K"; TEXT="${5:-paused after the save $(basename "$K")}" ;;
  *) CKPT="$DIR/ckpt/$(basename "$DIR")_${K}k.pth"; TEXT="${5:-paused after the ${K}k save}" ;;
esac
OPT="${CKPT%.pth}_optimizer.pth"
END=$((SECONDS + MAX_WAIT))
echo "$(date -u +%FT%TZ) waits for $CKPT, then stops $PID and writes $DIR/stop"
while [ "$SECONDS" -lt "$END" ]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) trainer $PID ended before the save: nothing done"; exit 2
  fi
  if [ -s "$CKPT" ] && [ -s "$OPT" ] && python3 - "$CKPT" >/dev/null 2>&1 <<'PY'
import sys, torch
path = sys.argv[1]
torch.load(path, map_location="cpu", weights_only=False)
torch.load(path[:-4] + "_optimizer.pth", map_location="cpu", weights_only=False)
PY
  then
    echo "$(date -u +%FT%TZ) $CKPT and its optimizer load: kill $PID"
    kill "$PID"
    echo "$TEXT" > "$DIR/stop"
    echo "$(date -u +%FT%TZ) wrote $DIR/stop"; exit 0
  fi
  sleep 15
done
echo "$(date -u +%FT%TZ) no save in $MAX_WAIT s: nothing done"; exit 1
