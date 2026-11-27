#!/bin/bash
# Pull one ablation run of #482 from its box to elisa every 10 min, and copy
# its checkpoints to S3 from the box. pull_1126_stops_box.sh of #425 with the
# elisa folders of #482 (CF), with one change: the uplink of elisa is too
# slow for the checkpoints (about 0.16 MB/s on 11-27), so elisa sends no .pth
# bytes. At each poll, s3_box_copy.py copy sends from the box to
# s3://model.artifacts/manual/cf_runs/cf-482/<run>/ckpt/ the backbone of each
# stop (STOPS) and the optimizer file of the newest stop, with a presigned
# URL. The ETag of each object must equal the ETag of the box file. Only the
# optimizer file of the newest stop stays in S3: with its backbone it is the
# resume pair of a new box (EXECUTION_LOG.md). The record
# <run>/s3_box_record.tsv names the files in S3.
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
H="${BOX_H:-/workspace/hp64}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
XS="ssh -p $BOX_PORT $OPTS"
CKPT=$CF/$RUN/ckpt; MIRROR=$FF/results_box_$SHORT
log() { echo "$(date -u +%FT%TZ) $*"; }

end=$(( $(date +%s) + MAX_HOURS * 3600 ))
mkdir -p "$MIRROR" "$CF/$RUN/logs_box" "$CKPT" "$FF/results" "$FF/heads/eval"
log "start: $RUN from $LABEL ($ID), stops $STOPS"
while [ "$(date +%s)" -lt "$end" ] && [ ! -e "$CF/logs/stop_pull_$SHORT" ]; do
  timeout 1200 rsync -a --partial --exclude='*.pth' -e "$XS" "$BOX:$H/$RUN/ckpt/" "$CKPT/" < /dev/null \
    || log "pull of ckpt failed (rc $?)"
  timeout 3600 "$PY" "$HERE/s3_box_copy.py" copy "$RUN" "$BOX" "$BOX_PORT" --stops "$STOPS" \
    --box-h "$H" --record "$CF/$RUN/s3_box_record.tsv" || log "S3 copy failed (rc $?); next poll"
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
