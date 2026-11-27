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
# The script stops with exit code 2 at the first failure: a failed copy, an
# md5 mismatch, or a check of setup_hp64_box.sh that does not pass. Its last
# step writes the file setup_done_<run> in the box folder. start_box.sh and
# s3_box_copy.py seed refuse a box without this file. Run it on a new box only:
# the md5 check of the run folder fails when the box holds files of the run.
# The box keeps the layout of #425 (/workspace/hp64), so the follower and the
# health check of #425 work on it unchanged.
#   BOX=root@<ip> BOX_PORT=<port> CODE=code_<sha> RUN=<run> [HEAD_CODE=ff_code_<sha>] \
#     bash setup_box.sh >> $CF/logs/setup_<label>.log 2>&1
set -uo pipefail
BOX="${BOX:?}"; PORT="${BOX_PORT:?}"; CODE="${CODE:?}"; RUN="${RUN:?}"
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
HP="${HP64:-/home/jupyter/cf_runs/hp64}"
H="${BOX_H:-/workspace/hp64}"
FOLLOW=start_follow_w320e6f6_box.sh
HEADS=$HP/bench_1123/train_heads_tf32.py
HEADS_MD5="${HEADS_MD5:-6d3a59a1}"  # the TF32 head trainer of the reference scores (PR #474)
TOKEN="${TOKEN:-$CF/secrets/hf_token_agent.txt}"
HEAD_CODE="${HEAD_CODE:-}"
DONE="$H/setup_done_$RUN"
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
log() { echo "$(date -u +%FT%TZ) $*"; }
fail() { log "ABORT: $*"; exit 2; }
x() { timeout 600 ssh -n -p "$PORT" $OPTS "$BOX" "$@"; }
sums() { find . -type f ! -path "*/__pycache__/*" -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum | cut -c1-32; }

base_setup() {  # setup_hp64_box.sh of #425, then each of its checks
  local out rc want
  out=$(BOX=$BOX BOX_PORT=$PORT bash "$HP/box/setup_hp64_box.sh" 2>&1); rc=$?
  echo "$out"
  [ "$rc" = 0 ] || fail "setup_hp64_box.sh ended with exit code $rc"
  for want in "head code md5 match" "pip rc 0" "gift-eval rc 0" "GIFT-Eval data md5 match"; do
    grep -qF "$want" <<< "$out" || fail "setup_hp64_box.sh gave no line '$want'"
  done
}

block_keys() {  # the state dict keys of the TimesFM block of the code folder $1
  (cd "$1" && PYTHONDONTWRITEBYTECODE=1 python3 -c \
    'from src.encoders import create_encoder as c; print(sorted(c("timesfm", 22, 32).state_dict()))' 2>/dev/null)
}

copy() {  # copy $1 of elisa to $2 on the box, then compare their md5
  local a b shown
  if [ -d "$1" ]; then
    timeout 3600 rsync -a --exclude '__pycache__' -e "ssh -p $PORT $OPTS" "$1/" "$BOX:$2/" < /dev/null \
      || fail "rsync of $1 failed"
    a=$(cd "$1" && sums)
    b=$(x "cd $2 && find . -type f ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum" | cut -c1-32)
  else
    timeout 300 rsync -a -e "ssh -p $PORT $OPTS" "$1" "$BOX:$2" < /dev/null || fail "rsync of $1 failed"
    a=$(md5sum < "$1" | cut -c1-32)
    b=$(x "md5sum < $2" | cut -c1-32)
  fi
  case "$2" in *hf_token.txt) shown="(token)" ;; *) shown="$a $b" ;; esac  # no token hash in a log
  [ -n "$b" ] && [ "$a" = "$b" ] || fail "$2 md5 MISMATCH $shown"
  log "$2 md5 match ${shown%% *}"
}

[ -d "$CF/$CODE" ] && [ -d "$CF/box_train/$RUN" ] || fail "no $CF/$CODE or $CF/box_train/$RUN"
[ -s "$TOKEN" ] || fail "no token in $TOKEN"
[ -z "$HEAD_CODE" ] || [ -d "$CF/$HEAD_CODE" ] || fail "no $CF/$HEAD_CODE"
# The score code of #425 cannot build these two backbones (rows 6 and 8).
if grep -qE -- '--encoder-type timesfm|--depthwise-conv 0$' "$CF/box_train/$RUN/run_$RUN.sh" && [ -z "$HEAD_CODE" ]; then
  fail "$RUN needs the patched score code: HEAD_CODE=ff_code_<sha> (make_head_code.sh)"
fi
# Row 6: the score code must build the TimesFM block of the run code.
if grep -qE -- '--encoder-type timesfm' "$CF/box_train/$RUN/run_$RUN.sh"; then
  keys=$(block_keys "$CF/$CODE")
  [ -n "$keys" ] && [ "$keys" = "$(block_keys "$CF/$HEAD_CODE")" ] \
    || fail "the TimesFM block of $HEAD_CODE is not the block of $CODE"
fi
[ "$(md5sum < "$HEADS" | cut -c1-8)" = "$HEADS_MD5" ] || fail "$HEADS is not md5 $HEADS_MD5"
x "rm -f $DONE" || fail "no ssh to $BOX"
x "command -v curl" > /dev/null || fail "no curl on the box: the S3 copies of pull_box.sh need it"
base_setup
copy "$CF/$CODE" "$H/$CODE"
copy "$CF/box_train/$RUN" "$H/$RUN"
copy "$HP/box/$FOLLOW" "$H/ff/$FOLLOW"
copy "$HEADS" "$H/ff/train_heads_tf32.py"
[ -z "$HEAD_CODE" ] || copy "$CF/$HEAD_CODE" "$H/ff/code"
copy "$TOKEN" "$H/ff/code/experiments/hf_token.txt"
x "echo 'run $RUN code $CODE head_code ${HEAD_CODE:-hp64/ff/code} $(date -u +%FT%TZ)' > $DONE" \
  || fail "cannot write $DONE"
log "setup_box end: $DONE written"
