#!/bin/bash
# #373 round 3 — stop the meter once the queue is empty and every byte is here.
#
# `q_run.sh` logs `queue drained` and exits. Nothing else stops the box, and
# the budget guard only acts at the $5.50 floor, so a queue that drains with
# $12 left would burn all of it at $0.8144/h before anything noticed. This
# closes that gap.
#
# Usage: BOX_ID=<id> BOX_HOST=<h> BOX_PORT=<p> bash r3_reap.sh [poll seconds]
#
# Two gates, both of which must pass:
#
#   1. the queue holds no job that is queued or running;
#   2. for every file the box holds under /root/cf373_runs, the local copy
#      exists at exactly the remote size.
#
# Gate 2 is `ls`, not a log line. CLAUDE.md § Operational rules: an instance
# was destroyed in May 2026 while it still held the only copy of a resume
# bundle. A sync log that says nothing is not a sync that worked.
#
# `touch results/HOLD_BOX` keeps the box alive whatever the gates say.
#
# The vast.ai account is shared with other agent sessions, so the destroy
# names BOTH the id and the label this study set, and it refuses if the
# instance the id resolves to does not carry that label.
set -uo pipefail

POLL="${1:-300}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
RES="$STUDY/results"
Q="$HERE/q_queue.tsv"
STATE="$RES/queue"
R3="${CF373_R3:-/home/jupyter/cf373_r3/sync}"
BOX_ID="${BOX_ID:?BOX_ID}"
BOX_HOST="${BOX_HOST:?BOX_HOST}"
BOX_PORT="${BOX_PORT:?BOX_PORT}"
BOX_LABEL="${BOX_LABEL:-cf373-dual}"
VDIR="${VDIR:-/home/jupyter/wt-cf-373-run2}"
export PATH="$HOME/.local/bin:$PATH"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20)

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [reap] $*" | tee -a "$RES/r3_reap.log"; }

# Jobs that still need the BOX. An eval does not: it runs `--device cpu` on
# elisa's cores, off elisa's own copies of the backbone and the head. Counting
# evals here keeps a card rented through the eval tail — 19 evals at three
# concurrent slots, the last of them hours after the last head — at $0.8144/h
# for a machine no job can use. Gate 2 still holds every byte back before the
# destroy, so the artefacts are safe whatever this counts.
queue_left(){
  local n=0 id
  for id in $(awk -F'\t' '$1 !~ /^#/ && NF && $2 != "eval" {print $1}' "$Q"); do
    case "$(cat "$STATE/$id.state" 2>/dev/null || echo queued)" in
      queued|running) n=$(( n + 1 ));;
    esac
  done; echo "$n"
}

log "start poll=${POLL}s box=$BOX_ID ($BOX_LABEL) root=$R3"
while :; do
  sleep "$POLL"

  [ -f "$RES/HOLD_BOX" ] && { log "held by results/HOLD_BOX"; continue; }
  [ -f "$RES/r3_reaped" ] && { log "already reaped — standing down"; exit 0; }

  n="$(queue_left)"
  [ "$n" -eq 0 ] || continue
  log "queue drained; checking every remote byte is here"

  # The instance must still be ours, by id AND by label.
  row="$( (cd "$VDIR" && timeout 120 vastrun-status 2>/dev/null) \
          | awk -v b="$BOX_ID" '$1==b')"
  if [ -z "$row" ]; then
    log "box $BOX_ID is not in vastrun-status — nothing to destroy"
    touch "$RES/r3_reaped"; exit 0
  fi
  case "$row" in
    *"$BOX_LABEL"*) ;;
    *) log "REFUSING: id $BOX_ID does not carry label $BOX_LABEL — row: $row"
       continue;;
  esac

  missing=0; nfiles=0
  while read -r size path; do
    [ -n "${path:-}" ] || continue
    dst="$R3/${path#/root/cf373_runs/}"
    nfiles=$(( nfiles + 1 ))
    if [ ! -f "$dst" ] || [ "$(wc -c <"$dst")" != "$size" ]; then
      log "NOT YET: ${path#/root/cf373_runs/} is not here at $size B"
      missing=1
    fi
  done < <(ssh "${SSH_OPTS[@]}" -n -p "$BOX_PORT" "root@$BOX_HOST" \
           "find /root/cf373_runs -type f -printf '%s %p\n' 2>/dev/null" 2>/dev/null)

  if [ "$nfiles" -eq 0 ]; then
    log "the box lists no artefact — leaving it alone rather than guessing"
    continue
  fi
  [ "$missing" -eq 0 ] || { log "waiting for the sync loop"; continue; }

  log "all $nfiles file(s) verified local — destroying $BOX_ID ($BOX_LABEL)"
  out=$( (cd "$VDIR" && timeout 300 vastrun-destroy "$BOX_ID" "$BOX_LABEL") 2>&1 )
  if grep -qi "no marker" <<<"$out"; then
    # provision_box.sh adopts a box after its own SSH probe, and the kit
    # writes its ownership marker before that, so an adopted box carries
    # none. The id and the label both match this study's own box and every
    # artefact is here, so --force is what is left.
    log "no vastrun marker (adopted box) — destroying with --force"
    out=$( (cd "$VDIR" && timeout 300 vastrun-destroy "$BOX_ID" --force) 2>&1 )
  fi
  printf '%s\n' "$out" | sed 's/^/    /' | tee -a "$RES/r3_reap.log" >/dev/null
  touch "$RES/r3_reaped"
  log "reaped — the meter is off"
  exit 0
done
