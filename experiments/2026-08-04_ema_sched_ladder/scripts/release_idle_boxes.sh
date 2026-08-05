#!/bin/bash
# #393 — give a vast.ai box back when it has no work left.
#
# Usage:  bash scripts/release_idle_boxes.sh            # report only
#         RELEASE=1 bash scripts/release_idle_boxes.sh  # report and destroy
#
# A box costs $0.36 to $0.49 an hour whether or not anything is on its GPU.
# Since GIFT-Eval moved to elisa a box's cell spends hours with nothing to
# run, but that is a WAITING cell, not a finished one, and the box has to
# stay. What ends a box is its ladder driver exiting: at that point the
# cell has either reached the HOLD_ABOVE ceiling or stopped on the rule,
# and nothing on the box will start again by itself.
#
# Five gates, all of which must pass. Destroying is not reversible and the
# thing destroyed may hold the only copy of a checkpoint.
#
#   1. LABEL. The box is in results/vast_contracts.txt. Both the contract
#      ID and the label go to vastrun-destroy, so a stale row cannot take
#      out another session's instance. Vast.ai is a shared account.
#   2. AGREEMENT. vastrun-status still reports that ID under that label.
#   3. QUIET. No ladder.py, no train.py, no GIFT-Eval, and no unanswered
#      EVAL_REQUEST on the box.
#
#      With DRAIN=1 the last of those relaxes, and only that one. An
#      unanswered EVAL_REQUEST means the box's driver is blocked waiting
#      for a score — but since the eval moved to elisa, the box is not
#      producing that score and nothing on it will run until the score
#      arrives. The GPU is rented and idle for the whole of it. The
#      request is only tolerated when the eval no longer needs the box:
#      the broker has already pulled that stop's backbone and head here,
#      or has already written its score. Then destroying the box cannot
#      cost a measurement, because the measurement is running on local
#      files. A request whose checkpoints are NOT here still counts as
#      busy under DRAIN — releasing there would strand a head that cost
#      30,000 GPU steps.
#
#      The score itself no longer needs the box either. It used to: the
#      broker pushed it back and the box's ladder.py wrote the row. Since
#      scripts/scores_from_evals.py the pooled table is rebuilt from
#      elisa's own eval directories, so a score survives the box that
#      asked for it. What does NOT survive is the box's extend-rule
#      decision row, which is why every DRAIN release is recorded as a
#      budget stop in decisions.csv rather than an extend-rule branch.
#   4. BUNDLE. Every cell the claims file gives this box has its resume
#      bundle on elisa — newest backbone, its optimizer companion, the
#      losses CSV and the run log. CLAUDE.md's rule after the May 3 loss:
#      check for a resume bundle BEFORE destroying, and if the work is not
#      home, it does not get destroyed.
#   5. TWICE. The box read as quiet on the previous pass too. A box between
#      two legs is quiet for a few seconds.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
CONTRACTS="$EXP/results/vast_contracts.txt"
CLAIMS="$EXP/results/cell_claims.txt"
STATE="${RELEASE_STATE:-/tmp/cf393_idle.state}"
# Where the broker keeps the checkpoints it pulled for each eval, and where
# scores_from_evals.py reads the scores back out. leg_paths.sh owns the
# path; this only needs the root.
RUNS_ROOT="${CF393_RUNS:-/home/jupyter/checkpoints_backup/cf-393}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=20 -o BatchMode=yes)

log() { echo "[$(date '+%m-%d %H:%M:%S')] [release] $*"; }
touch "$STATE" 2>/dev/null

[ -f "$CONTRACTS" ] || { log "ABORT: no $CONTRACTS"; exit 2; }
[ -f "$CLAIMS" ]    || { log "ABORT: no $CLAIMS"; exit 2; }

STATUS="$(vastrun-status 2>/dev/null)" || { log "ABORT: vastrun-status failed"; exit 2; }

# The local sync root for a box, as launch_sync.sh lays them out.
sync_root() {  # <label>
  case "$1" in
    a) echo "$HOME/cf393_sync/2026-08-04_ema_sched_ladder" ;;
    *) echo "$HOME/cf393_sync_$1/2026-08-04_ema_sched_ladder" ;;
  esac
}

# Is every cell this box owns home? Prints what is missing, empty if none.
bundle_gaps() {  # <label>
  local lbl="$1" root cell dir newest base gaps=""
  root="$(sync_root "$lbl")"
  for cell in $(awk -v m="vast${lbl^^}" '$2==m {print $1}' "$CLAIMS"); do
    dir="$root/sync/$cell"
    newest="$(ls "$dir"/leg_*/cf393_"$cell"*_[0-9]*k.pth 2>/dev/null \
              | sed -E 's|.*_([0-9]+)k\.pth$|\1 &|' | sort -k1,1n | tail -1 \
              | cut -d' ' -f2-)"
    if [ -z "$newest" ]; then gaps+="$cell:no-backbone "; continue; fi
    base="${newest%.pth}"
    [ -s "${base}_optimizer.pth" ] || gaps+="$cell:no-optimizer "
    ls "$(dirname "$newest")"/*_losses.csv >/dev/null 2>&1 || gaps+="$cell:no-losses "
    [ -s "$root/results/run_cf393_$cell.log" ] || gaps+="$cell:no-runlog "
  done
  printf '%s' "$gaps"
}

# DRAIN: is every reason this box reads busy an eval that elisa is already
# holding the checkpoints for? Anything else — a driver, a training, a
# GIFT-Eval on the box itself — is real work and answers no.
#
# `busy` is the probe's output: a count, then one word per reason, e.g.
# "1 waiting:bb100k_teacher". Only `waiting:<stop dir>` can be drained, and
# only when the broker's local working directory for that stop holds the
# two checkpoints it evaluates, or the score is already written.
drainable() {  # <label> <busy string>
  local lbl="$1" reason wdir ok waits=0
  set -- $2 ; shift            # drop the leading count, keep the reasons
  [ $# -gt 0 ] || return 1
  for reason in "$@"; do
    case "$reason" in
      waiting:*) waits=$(( waits + 1 )) ;;
      # The driver process is what the waiting is done BY: eval_stop.sh
      # blocks inside it in a sleep loop until the score file appears. So
      # `driver` alongside a drainable `waiting:` is that same idleness
      # counted twice, not separate work. `driver` on its own is a cell
      # between legs and about to start training — never drainable.
      driver) continue ;;
      *) return 1 ;;               # train / gift-eval — real work
    esac
    ok=0
    for wdir in "$RUNS_ROOT/_broker/$lbl"/*/"${reason#waiting:}"; do
      [ -d "$wdir" ] || continue
      if [ -s "$wdir/score.txt" ]; then ok=1; break; fi
      if [ -s "$wdir/backbone.pth" ] && [ -s "$wdir/head.pth" ]; then ok=1; break; fi
    done
    [ "$ok" -eq 1 ] || { log "$lbl: $reason has no checkpoints here — not drainable"; return 1; }
  done
  [ "$waits" -gt 0 ]
}

# A released box takes its ladder driver with it, so no extend-rule branch
# is ever recorded for the stop it was sitting on. Write the stop down as
# what it actually was: the spend order ended the cell, not the rule. The
# branch value is its own, never one of the rule's, so the report cannot
# confuse "stopped because both heads got worse" with "stopped because the
# credit ran out". merge_pooled.sh keys decisions on (cell, stop, branch),
# so this row sits beside any rule branch the same stop already carries.
record_budget_stop() {  # <label>
  local lbl="$1" cell stop dec="$EXP/results/decisions.csv"
  [ -f "$dec" ] || printf 'cell,stop,branch,extend,heads_next\n' > "$dec"
  for cell in $(awk -v m="vast${lbl^^}" '$2==m {print $1}' "$CLAIMS"); do
    stop=$(awk -F, -v c="$cell" '$1==c {print $4+0}' \
             "$EXP/results/ladder_all.csv" 2>/dev/null | sort -n | tail -1)
    [ -n "$stop" ] && [ "$stop" != "0" ] || stop=""
    [ -n "$stop" ] || { log "$lbl: $cell has no scored stop — no budget row written"; continue; }
    if grep -q "^$cell,$stop,budget_stop," "$dec" 2>/dev/null; then continue; fi
    printf '%s,%s,budget_stop,0,\n' "$cell" "$stop" >> "$dec"
    log "$lbl: recorded $cell bb$((stop/1000))k as branch=budget_stop"
  done
}

released=0
while read -r lbl cid host port rate who; do
  [ -n "${lbl:-}" ] || continue
  case "$lbl" in \#*) continue ;; esac

  # 2. agreement between the file and the account
  if ! grep -q "^$cid  *cf393-$lbl  *running" <<<"$STATUS"; then
    if grep -q "$cid" <<<"$STATUS"; then
      log "$lbl: contract $cid is not running as cf393-$lbl — leaving it alone"
    fi
    continue
  fi

  # 3. quiet?
  #
  # The bracketed first letters are not decoration. `pgrep -f` scans every
  # command line including the shell running this probe, and that shell's
  # command line contains these very patterns — so the plain form reports
  # three busy processes on a completely idle box, and no box is ever
  # released. `[l]adder` matches "ladder" but not the literal "[l]adder"
  # in the probe's own argv. `pgrep -a` prints what matched, so a false
  # positive is visible rather than a bare count.
  busy="$(timeout 60 ssh -n "${SSH_OPTS[@]}" -p "$port" "root@$host" '
    shopt -s nullglob
    n=0; why=""
    pgrep -f "[l]adder\.py --cells" >/dev/null && { n=$((n+1)); why="$why driver"; }
    pgrep -f "[t]rain\.py|[t]rain_forecasting_head\.py|[t]rain_ssl\.py" >/dev/null \
      && { n=$((n+1)); why="$why train"; }
    pgrep -f "[e]val_gift_eval_official\.py" >/dev/null && { n=$((n+1)); why="$why gift-eval"; }
    for r in /root/cf393_runs/*/eval/*/EVAL_REQUEST; do
      d="$(dirname "$r")"
      [ -f "$d/EVAL_DONE" ] || { n=$((n+1)); why="$why waiting:$(basename "$d")"; }
    done
    echo "$n$why"' 2>/dev/null)"
  if [ -z "$busy" ]; then
    log "$lbl: unreachable — leaving it alone"
    continue
  fi
  # "0" alone is idle; anything else is a count followed by what was found.
  if [ "$busy" != "0" ]; then
    if [ -n "${DRAIN:-}" ] && drainable "$lbl" "$busy"; then
      log "$lbl: DRAIN — ${busy#[0-9]}; every eval it waits on is running here"
    else
      log "$lbl: busy —${busy#[0-9]}"
      sed -i "/^$lbl /d" "$STATE" 2>/dev/null
      continue
    fi
  fi

  # 5. twice
  if ! grep -q "^$lbl " "$STATE" 2>/dev/null; then
    echo "$lbl $(date +%s)" >> "$STATE"
    log "$lbl: reads idle — will confirm on the next pass before releasing"
    continue
  fi

  # 4. bundle
  gaps="$(bundle_gaps "$lbl")"
  if [ -n "$gaps" ]; then
    log "$lbl: IDLE but NOT released — resume bundle incomplete: $gaps"
    continue
  fi

  log "$lbl: idle twice, bundle complete, cells=$(awk -v m="vast${lbl^^}" '$2==m{printf "%s ",$1}' "$CLAIMS")"
  if [ -z "${RELEASE:-}" ]; then
    log "$lbl: would destroy $cid (cf393-$lbl, \$$rate/h) — set RELEASE=1 to do it"
    continue
  fi
  log "$lbl: destroying $cid cf393-$lbl (\$$rate/h, provisioned by $who)"
  # </dev/null, and it matters. The loop below reads the contracts file on
  # STDIN, and vastrun-destroy reads stdin too — so without this it swallows
  # the rest of the file and the pass ends after the first box it destroys.
  # Silent: the log just stops, and every later box reads as "not visited"
  # rather than "left running". Boxes B and C each cost a whole extra pass.
  if vastrun-destroy "$cid" "cf393-$lbl" >/dev/null 2>&1 </dev/null; then
    log "$lbl: RELEASED $cid"
    record_budget_stop "$lbl"
    sed -i "/^$lbl /d" "$STATE" 2>/dev/null
    released=$(( released + 1 ))
  else
    log "$lbl: vastrun-destroy failed — box left running"
  fi
done <"$CONTRACTS"

[ "$released" -gt 0 ] && log "released $released box(es); rerun scripts/status.sh"
exit 0
