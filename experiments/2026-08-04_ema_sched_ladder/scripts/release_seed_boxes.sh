#!/bin/bash
# #393 — give a replicate-seed box back the moment its queue is empty.
#
# Usage:  bash scripts/release_seed_boxes.sh          # report only
#         RELEASE=1 bash scripts/release_seed_boxes.sh
#         RELEASE=1 LOOP=1 bash scripts/release_seed_boxes.sh   # every 10 min
#
# The gates, in order. Any one of them failing leaves the box running: a box
# costs $0.36/h and a lost head costs 30,000 GPU steps, so the asymmetry is
# not close.
#
#   1. ALLOWLIST. The box is a row of results/seed_boxes.txt, and BOTH its
#      contract ID and its label go to vastrun-destroy. Vast.ai is a shared
#      account across concurrent agent sessions (CLAUDE.md); nothing outside
#      this file is ever touched, and a label that does not start with
#      `cf393seed` is refused even if the file says otherwise.
#   2. QUEUE EMPTY. Every (cell, head, seed) the box was given has a score
#      on the box.
#   3. WORK IS HOME. Each of those scores is also on elisa's durable root,
#      and each carries its 97-config all_results.csv and its summary.txt
#      here — the evidence audit_scores.py checks, not just the number.
#   4. NOTHING RUNNING. No head training and no eval process on the box.
#
# Gate 3 is the one that matters. The score is 7 bytes and the sync loop's
# tick is 15 minutes; destroying on the box's own word would throw away the
# evidence behind a number the report publishes.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
RES="$EXP/results"
RUNS="${RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
BOXES="$RES/seed_boxes.txt"
LOG="$RES/seed_release.log"
WT_ROOT="${WT:-/tmp/contrastive-forecasting-393}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)
PROTOCOL_SEED=20260722

log(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] [release] $*" | tee -a "$LOG"; }

pass(){
  local lbl cid host port gpu cores rate cell seeds
  while read -r lbl cid host port gpu cores rate cell seeds; do
    case "$lbl" in ''|\#*) continue ;; esac

    # 1. allowlist — the label this row claims must be one of ours.
    local vlabel
    vlabel="$(timeout 90 vastrun-status 2>/dev/null \
              | awk -v id="$cid" '$1==id{print $2}')"
    if [ -z "$vlabel" ]; then
      log "$lbl: contract $cid not running — nothing to release"
      continue
    fi
    case "$vlabel" in
      cf393seed*) ;;
      *) log "$lbl: REFUSING — contract $cid carries label '$vlabel', not a cf393seed box"
         continue ;;
    esac

    # 2 + 3. every job home, with its evidence.
    local missing=0 n=0 s enc score
    for s in ${seeds//,/ }; do
      for enc in student teacher; do
        n=$(( n + 1 ))
        local sfx=""; [ "$s" != "$PROTOCOL_SEED" ] && sfx="_s$s"
        score="$RUNS/$cell/eval/score_bb100k_${enc}${sfx}.txt"
        [ -s "$score" ] || { missing=$(( missing + 1 )); continue; }
        local ev="$RES/eval/$cell/eval/bb100k_${enc}${sfx}/gift"
        [ -s "$ev/summary.txt" ] || { missing=$(( missing + 1 )); continue; }
        # 97 data rows, not just a file that exists.
        [ "$(( $(wc -l < "$ev/all_results.csv" 2>/dev/null || echo 0) - 1 ))" \
          -eq 97 ] || missing=$(( missing + 1 ))
      done
    done
    if [ "$missing" -gt 0 ]; then
      log "$lbl ($cell): $(( n - missing ))/$n home with evidence — keeping"
      continue
    fi

    # 4. nothing still running there.
    local busy
    busy="$(timeout 60 ssh -n "${SSH_OPTS[@]}" -p "$port" "root@$host" \
      'pgrep -cf "train_forecasting_head|eval_gift_eval_official" 2>/dev/null || echo 0' \
      2>/dev/null)"
    if [ -z "$busy" ]; then
      log "$lbl: unreachable — keeping (a box that cannot be asked is not idle)"
      continue
    fi
    if [ "$busy" -gt 0 ]; then
      log "$lbl ($cell): $n/$n home but $busy process(es) still running — keeping"
      continue
    fi

    if [ -z "${RELEASE:-}" ]; then
      log "$lbl ($cell): would destroy $cid $vlabel (\$$rate/h) — set RELEASE=1"
      continue
    fi
    log "$lbl ($cell): $n/$n home with evidence, idle — destroying $cid $vlabel"
    if ( cd "$WT_ROOT" && vastrun-destroy "$cid" "$vlabel" >/dev/null 2>&1 </dev/null ); then
      log "$lbl: destroyed"
      # Mark the roster so the next pass, and a reader, both know.
      sed -i "s|^${lbl}  |# released ${lbl}  |" "$BOXES"
    else
      log "$lbl: vastrun-destroy failed — box left running"
    fi
  done < "$BOXES"
}

[ -f "$BOXES" ] || { log "ABORT: no roster at $BOXES"; exit 2; }
while :; do
  pass
  [ -n "${LOOP:-}" ] || break
  sleep "${RELEASE_POLL:-600}"
done
