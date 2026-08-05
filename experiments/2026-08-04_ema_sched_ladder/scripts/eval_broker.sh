#!/bin/bash
# #393 — run the vast.ai boxes' GIFT-Evals on elisa.
#
# Usage:  bash scripts/eval_broker.sh            # loop until killed
#         BROKER_ONCE=1 bash scripts/eval_broker.sh   # one pass, for checks
#
# The split decided on PR #394: a cell's backbone and its quantile heads
# train on a rented GPU, and the GIFT-Eval that scores them runs here. The
# eval is CPU-bound — one core, 696 MiB of VRAM, a GPU 30% busy — and a
# vast box is in Exclusive_Process mode, so an eval there holds the box's
# only CUDA context and the GPU idles under it for hours.
#
# The direction of travel is elisa -> vast, always. A vast container has no
# route back here and no key for it, so a box cannot push its request: it
# leaves one on its own disk and this loop comes and finds it.
#
#   vast   eval_stop.sh trains the head, writes <eval dir>/EVAL_REQUEST,
#          then waits for its score file to appear.
#   elisa  this loop reads the request, pulls the two checkpoints (5.2 MB
#          backbone, 0.45 MB head), runs eval_local.sh under the eval-slot
#          cap, and pushes summary.txt, all_results.csv and the score back.
#   vast   the waiting eval_stop.sh sees the score and the ladder carries on.
#
# The score file is pushed LAST and through a temporary name, because its
# appearance is the signal the waiter is watching for. A half-written score
# would be read as a number.
#
# What stops two passes dispatching one eval is the LOCAL lock a worker
# holds for its whole life, not the EVAL_TAKEN marker on the box. That
# distinction is the difference between a retry and a strand: a marker
# outlives the worker that set it, so a broker killed mid-eval would leave
# every request it had claimed looking busy forever, and the heads waiting
# on them had already cost 15,000 GPU steps each. A flock dies with its
# holder, so the next tick simply picks the request up again. EVAL_TAKEN
# stays for the person reading the box, and carries no authority.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/leg_paths.sh"

WT="${WT:-/tmp/contrastive-forecasting-393}"
ROOT="$(runs_root)" || exit 2
BROKER="$ROOT/_broker"
BOXES_FILE="${BROKER_BOXES:-$(dirname "$HERE")/results/broker_boxes.txt}"
POLL="${BROKER_POLL:-120}"
# Ceiling on workers alive at once, waiting ones included. The eval-slot
# semaphore is the real cap on cores; this only stops an unbounded pile of
# waiters if every box finishes a head in the same tick.
MAX_WORKERS="${BROKER_MAX_WORKERS:-12}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)
SAFE_PULL="$(dirname "$(dirname "$HERE")")/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh"

mkdir -p "$BROKER/workers" || exit 2
log() { echo "[$(date '+%m-%d %H:%M:%S')] [broker] $*"; }

[ -f "$BOXES_FILE" ] || { log "ABORT: no box list at $BOXES_FILE"; exit 2; }
[ -f "$SAFE_PULL" ] || { log "ABORT: no safe_pull.sh at $SAFE_PULL"; exit 2; }

# Workers alive now: a worker holds an exclusive flock on its key file for
# its whole life, so this counts processes rather than leftover files.
workers_alive() {
  local n=0 f
  for f in "$BROKER/workers"/*.lock; do
    [ -f "$f" ] || continue
    flock -n "$f" true 2>/dev/null || n=$(( n + 1 ))
  done
  printf '%d\n' "$n"
}

# One request, end to end. Runs in its own process.
serve() {  # <label> <host> <port> <cell> <stop_k> <enc> <rbb> <rhead> <rout> <rscore>
  local lbl=$1 host=$2 port=$3 cell=$4 stop_k=$5 enc=$6
  local rbb=$7 rhead=$8 rout=$9 rscore=${10}
  local key="${lbl}_${cell}_bb${stop_k}k_${enc}"
  local lock="$BROKER/workers/$key.lock"

  exec 7>>"$lock" || return 1
  flock -n 7 || { log "$key already served by a live worker"; return 0; }

  local wdir="$BROKER/$lbl/$cell/bb${stop_k}k_${enc}"
  mkdir -p "$wdir"
  local bb="$wdir/backbone.pth" head="$wdir/head.pth"
  local score="$wdir/score.txt"

  # Floors match the classes sync_loop.sh measured on this run: backbone
  # 5.19 MB, quantile head 0.45 MB. A short pull here would be an eval of
  # a truncated checkpoint, which produces a number rather than an error.
  bash "$SAFE_PULL" "$host" "$port" "$rbb" "$bb" 3000000 >>"$wdir/pull.log" 2>&1 \
    || { log "$key: backbone pull failed"; unclaim "$host" "$port" "$rout"; return 1; }
  bash "$SAFE_PULL" "$host" "$port" "$rhead" "$head" 200000 >>"$wdir/pull.log" 2>&1 \
    || { log "$key: head pull failed"; unclaim "$host" "$port" "$rout"; return 1; }

  log "$key: checkpoints in, queueing for an eval slot"
  WT="$WT" RUNS="$ROOT" bash "$HERE/eval_local.sh" \
    "$cell" "$stop_k" "$enc" "$bb" "$head" "$wdir" "$score" >>"$wdir/worker.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$score" ]; then
    log "$key: eval_local.sh rc=$rc — leaving the request for the next tick"
    unclaim "$host" "$port" "$rout"
    return 1
  fi

  # Results first, the score last: the score appearing is what releases the
  # waiter on the box, so everything it might read has to be there already.
  scp "${SSH_OPTS[@]}" -P "$port" "$wdir/gift/summary.txt" \
      "root@$host:$rout/gift/summary.txt" >/dev/null 2>&1
  scp "${SSH_OPTS[@]}" -P "$port" "$wdir/gift/all_results.csv" \
      "root@$host:$rout/gift/all_results.csv" >/dev/null 2>&1
  scp "${SSH_OPTS[@]}" -P "$port" "$score" "root@$host:$rscore.incoming" \
      >/dev/null 2>&1 \
    || { log "$key: score upload failed"; unclaim "$host" "$port" "$rout"; return 1; }
  ssh -n "${SSH_OPTS[@]}" -p "$port" "root@$host" \
      "mv -f '$rscore.incoming' '$rscore' && touch '$rout/EVAL_DONE'" \
      >/dev/null 2>&1 \
    || { log "$key: score swap failed"; unclaim "$host" "$port" "$rout"; return 1; }

  log "$key: DONE score=$(cat "$score")"
  return 0
}

unclaim() {  # <host> <port> <remote eval dir>
  ssh -n "${SSH_OPTS[@]}" -p "$2" "root@$1" "rm -f '$3/EVAL_TAKEN'" >/dev/null 2>&1
}

pass() {
  local lbl host port raw
  while read -r lbl host port; do
    [ -n "${lbl:-}" ] || continue
    case "$lbl" in \#*) continue ;; esac

    # One ssh per box: list every unclaimed request and claim it in the same
    # breath, so two passes overlapping cannot both take one.
    raw="$(timeout 90 ssh -n "${SSH_OPTS[@]}" -p "$port" "root@$host" '
      shopt -s nullglob
      for r in /root/cf393_runs/*/eval/*/EVAL_REQUEST; do
        d="$(dirname "$r")"
        [ -f "$d/EVAL_DONE" ] && continue
        touch "$d/EVAL_TAKEN"
        echo "@@REQ"
        cat "$r"
      done' 2>/dev/null)" || continue
    [ -n "$raw" ] || continue

    local cell="" stop_k="" enc="" rbb="" rhead="" rout="" rscore=""
    while IFS= read -r line; do
      case "$line" in
        @@REQ)
          [ -n "$cell" ] && dispatch "$lbl" "$host" "$port" \
            "$cell" "$stop_k" "$enc" "$rbb" "$rhead" "$rout" "$rscore"
          cell=""; stop_k=""; enc=""; rbb=""; rhead=""; rout=""; rscore="" ;;
        cell=*)      cell="${line#cell=}" ;;
        stop_k=*)    stop_k="${line#stop_k=}" ;;
        enc=*)       enc="${line#enc=}" ;;
        backbone=*)  rbb="${line#backbone=}" ;;
        head=*)      rhead="${line#head=}" ;;
        out_dir=*)   rout="${line#out_dir=}" ;;
        score_out=*) rscore="${line#score_out=}" ;;
      esac
    done <<<"$raw"
    [ -n "$cell" ] && dispatch "$lbl" "$host" "$port" \
      "$cell" "$stop_k" "$enc" "$rbb" "$rhead" "$rout" "$rscore"
  done <"$BOXES_FILE"
}

dispatch() {  # <label> <host> <port> <cell> <stop_k> <enc> <rbb> <rhead> <rout> <rscore>
  local lbl=$1 cell=$4 stop_k=$5 enc=$6
  if [ -z "$cell" ] || [ -z "$stop_k" ] || [ -z "$enc" ] || [ -z "${10}" ]; then
    log "$lbl: malformed request (cell='$cell' stop='$stop_k' enc='$enc') — skipping"
    return
  fi
  # A request stays listed while its worker runs, so every tick sees it
  # again. Testing the worker's lock here rather than in serve() keeps the
  # log to one line per eval instead of one per tick.
  local lock="$BROKER/workers/${lbl}_${cell}_bb${stop_k}k_${enc}.lock"
  if [ -f "$lock" ] && ! flock -n "$lock" true 2>/dev/null; then
    return
  fi
  local alive; alive="$(workers_alive)"
  if [ "$alive" -ge "$MAX_WORKERS" ]; then
    log "$alive workers alive (max $MAX_WORKERS) — $lbl/$cell bb${stop_k}k $enc waits a tick"
    return
  fi
  log "take $lbl/$cell bb${stop_k}k $enc"
  ( serve "$@" ) &
}

log "start boxes=$BOXES_FILE poll=${POLL}s runs=$ROOT wt=$WT"
while :; do
  pass
  [ -n "${BROKER_ONCE:-}" ] && break
  sleep "$POLL"
done
log "one pass done (BROKER_ONCE)"
wait
