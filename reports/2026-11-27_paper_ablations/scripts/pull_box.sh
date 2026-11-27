#!/bin/bash
# Pull one ablation run of #482 from its box to elisa every 10 min, and send
# its backbones to S3. pull_1126_stops_box.sh of #425 with the elisa folders of
# #482 (CF) and two changes for the small disk of elisa:
#   - of the .pth files it pulls only the backbones of the stops (STOPS), with
#     no optimizer file.
#   - each pulled backbone that loads goes to s3://model.artifacts/manual/
#     cf_runs/cf-482/<run>/ckpt/ with upload_and_delete_pth.py (ETag check),
#     then leaves elisa. A pulled file that does not load (the trainer still
#     wrote it) is deleted and comes again at the next poll. The record
#     <run>/s3_record.tsv names the files in S3, and they do not come back.
# It also pulls the other files of ckpt/ (the losses CSV), the logs into
# <run>/logs_box/, the results folder of the box follower into
# ff/results_box_<short>/, and, for each new score of this run, its eval folder
# (with machine.txt) into ff/heads/eval/<tag>/ and then its score file into
# ff/results/. The file stop_pull_<short> in CF/logs ends it.
#   RUN=<run> SHORT=<short> LABEL=<label> ID=<vast id> BOX=root@<ip> BOX_PORT=<port> \
#     setsid -f nohup bash pull_box.sh >> $CF/logs/pull_<short>.log 2>&1 < /dev/null
set -uo pipefail
CF="${CF:-/home/jupyter/cf_runs/cf-482}"; FF=$CF/ff
RUN="${RUN:?}"; SHORT="${SHORT:?}"; LABEL="${LABEL:?}"; ID="${ID:?}"
BOX="${BOX:?}"; BOX_PORT="${BOX_PORT:?}"; MAX_HOURS="${MAX_HOURS:-48}"
STOPS="${STOPS:-10 25 50 75 100 125 150 166}"
PY="${PY:-/home/jupyter/rnd/.venv/bin/python3}"  # boto3
S3_TOOL=/home/jupyter/cf_runs/hp64/tools/upload_and_delete_pth.py
H=/workspace/hp64
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
XS="ssh -p $BOX_PORT $OPTS"
CKPT=$CF/$RUN/ckpt; MIRROR=$FF/results_box_$SHORT
log() { echo "$(date -u +%FT%TZ) $*"; }

backbones() {  # the rsync filter: the backbones of the stops, less the files in S3
  local f s
  KEEP=()
  for f in $(cut -f2 "$CF/$RUN/s3_record.tsv" 2>/dev/null); do KEEP+=(--exclude="$(basename "$f")"); done
  for s in $STOPS; do KEEP+=(--include="*_${s}k.pth"); done
}

to_s3() {  # each pulled backbone that loads goes to S3. One that does not load goes away
  local f
  for f in "$CKPT"/*.pth; do
    [ -e "$f" ] || continue
    "$PY" -c 'import sys, torch; torch.load(sys.argv[1], map_location="cpu", weights_only=True)' "$f" 2>/dev/null \
      || { log "$(basename "$f") does not load yet: deleted, next poll"; rm -f "$f"; }
  done
  ls "$CKPT"/*.pth >/dev/null 2>&1 || return 0
  "$PY" "$S3_TOOL" "$CKPT" "manual/cf_runs/cf-482/$RUN/ckpt" "$CF/$RUN/s3_record.tsv" 2>&1 | tail -n 1
}

end=$(( $(date +%s) + MAX_HOURS * 3600 ))
mkdir -p "$MIRROR" "$CF/$RUN/logs_box" "$CKPT" "$FF/results" "$FF/heads/eval"
log "start: $RUN from $LABEL ($ID), stops $STOPS"
while [ "$(date +%s)" -lt "$end" ] && [ ! -e "$CF/logs/stop_pull_$SHORT" ]; do
  backbones
  timeout 1200 rsync -a --partial "${KEEP[@]}" --exclude='*.pth' -e "$XS" "$BOX:$H/$RUN/ckpt/" "$CKPT/" < /dev/null \
    || log "pull of ckpt failed (rc $?)"
  to_s3
  timeout 600 rsync -a -e "$XS" "$BOX:$H/$RUN/logs/" "$CF/$RUN/logs_box/" < /dev/null \
    || log "pull of logs failed (rc $?)"
  timeout 600 rsync -a --exclude 'evalslots/' -e "$XS" "$BOX:$H/ff/results/" "$MIRROR/" < /dev/null \
    || log "mirror rsync failed (rc $?)"
  for s in "$MIRROR"/score_${RUN}_bb*.txt; do
    [ -s "$s" ] || continue
    name=$(basename "$s"); tag=${name#score_}; tag=${tag%.txt}
    [ -s "$FF/results/$name" ] && continue
    if timeout 900 rsync -a -e "$XS" "$BOX:$H/ff/heads/eval/$tag/" "$FF/heads/eval/$tag/" < /dev/null; then
      echo "trained and scored on the rented box $LABEL $ID, pulled $(date -u +%FT%TZ)" > "$FF/heads/eval/$tag/machine.txt"
      cp "$s" "$FF/results/$name.tmp" && mv "$FF/results/$name.tmp" "$FF/results/$name" && log "pulled $tag: $(cat "$s")"
    else
      log "pull of eval/$tag failed; next poll"
    fi
  done
  sleep 600
done
log "end"
