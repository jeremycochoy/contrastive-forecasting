#!/bin/bash
# Start one ablation run of #482 on its box (setup_box.sh done). This is
# start_1124.sh of #425 with KIND=full and the elisa puller of #482:
#   1. the supervisor of the run (it resumes from the last checkpoint that
#      loads after a crash of the trainer).
#   2. the head follower of #425 (start_follow_w320e6f6_box.sh, the TF32 head
#      trainer): beside the trainer, one head of 30k steps and its GM-MASE at
#      each stop of the reference, as for the reference scores.
#   3. the elisa puller pull_box.sh (scores, losses CSV, logs, and the
#      backbones of the stops to S3).
# It prints the box PIDs.
#   BOX=root@<ip> BOX_PORT=<port> RUN=<run> SHORT=<short> LABEL=<label> ID=<vast id> bash start_box.sh
set -euo pipefail
BOX="${BOX:?}"; PORT="${BOX_PORT:?}"; RUN="${RUN:?}"; SHORT="${SHORT:?}"; LABEL="${LABEL:?}"; ID="${ID:?}"
CF="${CF:-/home/jupyter/cf_runs/cf-482}"
STOPS="${STOPS:-10 25 50 75 100 125 150 166}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
H=/workspace/hp64
O="-o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
x() { timeout 120 ssh -n -p "$PORT" $O "$BOX" "$@"; }
log() { echo "$(date -u +%FT%TZ) $LABEL: $*"; }
[ "$(x "pgrep -c python3 || true")" = 0 ] || { log "ABORT: a python3 process runs on the box"; exit 2; }
x "cd $H/$RUN && BB_GPU=0 setsid -f nohup bash supervise_$RUN.sh >> logs/supervise.log 2>&1 < /dev/null"
log "supervisor started: $(x "pgrep -f '^bash supervise_$RUN.sh' | head -n 1")"
x "cd $H/ff && FF_TRAINER=$H/ff/train_heads_tf32.py E_RUNS=$RUN E_STOPS='$STOPS' E_LOG=follow_${SHORT}_box.log bash start_follow_w320e6f6_box.sh"
sleep 2; log "follower started: $(x "pgrep -f 'follow_multi.sh' | head -n 1")"
mkdir -p "$CF/logs"
RUN=$RUN SHORT=$SHORT LABEL=$LABEL ID=$ID BOX=$BOX BOX_PORT=$PORT STOPS="$STOPS" CF=$CF \
  setsid -f nohup bash "$HERE/pull_box.sh" >> "$CF/logs/pull_${SHORT}.log" 2>&1 < /dev/null
log "elisa puller started (log $CF/logs/pull_${SHORT}.log)"
