#!/bin/bash
# Set up one rented box for one ablation run of #482. This is setup_1124.sh of
# #425 with the elisa folders of #482 (CF) in place of hp64:
#   1. setup_hp64_box.sh of #425, unchanged: the head code (hp64/ff/code,
#      a416b3ae), the packages, gift-eval and the GIFT-Eval data.
#   2. The run code folder, the run folder, the follower start script of the
#      width-320 6 + 6 runs and the TF32 head trainer, each with an md5 check.
#   3. The read token of the card (cf-482/secrets) in the head code, in place
#      of the shared token near its quota. The run code folder holds it
#      already (make_code_folder.sh).
#   4. For the rows 6 and 8 only (HEAD_CODE=ff_code_<sha>, make_head_code.sh):
#      the score code with the #482 patch, in place of the head code.
# The box keeps the layout of #425 (/workspace/hp64), so the follower and the
# health check of #425 work on it unchanged.
#   BOX=root@<ip> BOX_PORT=<port> CODE=code_<sha> RUN=<run> [HEAD_CODE=ff_code_<sha>] \
#     bash setup_box.sh >> $CF/logs/setup_<label>.log 2>&1
set -uo pipefail
BOX="${BOX:?}"; PORT="${BOX_PORT:?}"; CODE="${CODE:?}"; RUN="${RUN:?}"
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
HP=/home/jupyter/cf_runs/hp64
H=/workspace/hp64
FOLLOW=start_follow_w320e6f6_box.sh
HEADS=$HP/bench_1123/train_heads_tf32.py
HEADS_MD5=6d3a59a1  # the TF32 head trainer of the reference scores (PR #474)
TOKEN="${TOKEN:-$CF/secrets/hf_token_agent.txt}"
HEAD_CODE="${HEAD_CODE:-}"
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
log() { echo "$(date -u +%FT%TZ) $*"; }

[ -d "$CF/$CODE" ] && [ -d "$CF/box_train/$RUN" ] || { log "ABORT: no $CF/$CODE or $CF/box_train/$RUN"; exit 2; }
[ -s "$TOKEN" ] || { log "ABORT: no token in $TOKEN"; exit 2; }
[ -z "$HEAD_CODE" ] || [ -d "$CF/$HEAD_CODE" ] || { log "ABORT: no $CF/$HEAD_CODE"; exit 2; }
# The score code of #425 cannot build these two backbones (rows 6 and 8).
if grep -qE -- '--encoder-type timesfm|--depthwise-conv 0$' "$CF/box_train/$RUN/run_$RUN.sh" && [ -z "$HEAD_CODE" ]; then
  log "ABORT: $RUN needs the patched score code: HEAD_CODE=ff_code_<sha> (make_head_code.sh)"; exit 2
fi
[ "$(md5sum < "$HEADS" | cut -c1-8)" = "$HEADS_MD5" ] || { log "ABORT: $HEADS is not md5 $HEADS_MD5"; exit 2; }
BOX=$BOX BOX_PORT=$PORT bash $HP/box/setup_hp64_box.sh
sums() { find . -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum | cut -c1-32; }
PAIRS=("$CF/$CODE $H/$CODE" "$CF/box_train/$RUN $H/$RUN" "$HP/box/$FOLLOW $H/ff/$FOLLOW"
       "$HEADS $H/ff/train_heads_tf32.py")
[ -z "$HEAD_CODE" ] || PAIRS+=("$CF/$HEAD_CODE $H/ff/code")
PAIRS+=("$TOKEN $H/ff/code/experiments/hf_token.txt")
for pair in "${PAIRS[@]}"; do
  set -- $pair
  if [ -d "$1" ]; then
    timeout 3600 rsync -a --exclude '__pycache__' -e "ssh -p $PORT $OPTS" "$1/" "$BOX:$2/" < /dev/null || log "rsync of $1 failed"
    a=$(cd "$1" && sums)
    b=$(timeout 600 ssh -n -p "$PORT" $OPTS "$BOX" "cd $2 && find . -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum" | cut -c1-32)
  else
    timeout 300 rsync -a -e "ssh -p $PORT $OPTS" "$1" "$BOX:$2" < /dev/null || log "rsync of $1 failed"
    a=$(md5sum < "$1" | cut -c1-32)
    b=$(timeout 300 ssh -n -p "$PORT" $OPTS "$BOX" "md5sum < $2" | cut -c1-32)
  fi
  case "$2" in *hf_token.txt) shown="(token)" ;; *) shown="$a $b" ;; esac  # no token hash in a log
  [ "$a" = "$b" ] && log "$2 md5 match ${shown%% *}" || log "$2 md5 MISMATCH $shown"
done
log "setup_box end"
