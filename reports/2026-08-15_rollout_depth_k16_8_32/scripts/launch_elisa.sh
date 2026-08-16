#!/bin/bash
# #401 — the launch plan on elisa. The card's stages, in the card's order,
# with the box's concurrency.
#
# `run.sh` runs the stages one after the other, which is the right default and
# is what the trial ran. This adds one thing, and only one: how many of them
# elisa runs at the same time.
#
#   phase 1   the three arms are three independent backbone ladders. The
#             measured peak is 5.9 GiB per leg (results/smoke_k16.csv) and
#             GPU 0 has 19.1 GiB free, so two legs fit and still leave the
#             7.0 GiB the head gate asks for. Two, therefore, not three: a
#             third leg fits the card but starves every head behind it.
#             The order is the card's, k = 16 first, then k = 8, then k = 32.
#
#   phase 2   no backbone runs, so the card is free. The two arms run as two
#             `phase2.sh` invocations with SEPARATE head locks, so one head
#             per arm trains at a time instead of one head in total.
#
# `results/SLOTS` holds the phase-1 arm count and is read fresh on every
# poll. Raising it starts the next arm without a restart.
#
# Everything below is idempotent, the same as the stages it calls. A machine
# that reboots mid-study loses the legs in flight and nothing else: re-run
# this script and every finished stop, head and eval is a no-op.
#
# Usage:  BB_GPU=0 nohup setsid bash scripts/launch_elisa.sh &
#         SLOTS=3 BB_GPU=0 bash scripts/launch_elisa.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/study.sh"

BB_GPU="${BB_GPU:-0}"
POLL="${POLL:-60}"
# Seconds between two arm starts. Both legs open the same streaming dataset,
# and starting them together puts two cold HF readers on one connection.
STAGGER="${STAGGER:-180}"
mkdir -p "$CF401_RESULTS"

SLOTS_FILE="$CF401_RESULTS/SLOTS"
STATE="$CF401_RESULTS/RUN_STATE.md"
LOG="$CF401_RESULTS/launch.log"
[ -f "$SLOTS_FILE" ] || echo "${SLOTS:-2}" >"$SLOTS_FILE"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] [#401 launch] $*" | tee -a "$LOG"; }

# The arm count, read fresh every poll. A file that says nothing usable is 2,
# never 0 — a 0 would park the study with no arm running and no error.
slots(){
  local n; n="$(tr -dc '0-9' <"$SLOTS_FILE" 2>/dev/null)"
  if [ -n "$n" ] && [ "$n" -ge 1 ]; then echo "$n"; else echo 2; fi
}

# What a re-dispatched session reads first. One file, overwritten, so it is
# never a log to scroll.
state(){  # <stage> <note>
  { echo "# #401 run state"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- stage: $1"
    echo "- note: $2"
    echo "- launcher pid: $$"
    echo "- gpu: $BB_GPU, arm slots: $(slots) (edit \`results/SLOTS\`)"
    echo "- root: \`$CF401_ROOT\`"
    echo
    echo "## Scores so far"
    echo
    echo '```'
    cat "$CF401_RESULTS/scores.csv" 2>/dev/null || echo "(none yet)"
    echo '```'
    echo
    echo "## Checkpoints on disk"
    echo
    echo '```'
    ls -1 "$CF401_ROOT"/k*/*/leg_*k/*k.pth 2>/dev/null \
      | grep -v optimizer | sed "s#$CF401_ROOT/##" || echo "(none yet)"
    echo '```'
  } >"$STATE.tmp" && mv -f "$STATE.tmp" "$STATE"
}

log "START gpu=$BB_GPU depths='$CF401_DEPTHS' stops='$CF401_STOPS' slots=$(slots)"
state "phase 1" "starting"

# ---- Phase 1 -----------------------------------------------------------------
pids=(); names=()
for k in $CF401_DEPTHS; do
  while [ "$(jobs -rp | wc -l)" -ge "$(slots)" ]; do
    state "phase 1" "k=$k waits for an arm slot"
    sleep "$POLL"
  done
  log "arm k=$k START (slots=$(slots))"
  DEPTHS="$k" BB_GPU="$BB_GPU" HEAD_BG=1 \
    nohup bash "$HERE/phase1.sh" >>"$CF401_RESULTS/phase1_k${k}.out" 2>&1 &
  pids+=($!); names+=("k=$k")
  state "phase 1" "arms started: ${names[*]}"
  sleep "$STAGGER"
done

p1_failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  log "arm ${names[$i]} rc=$rc"
  [ $rc -eq 0 ] || p1_failed=$(( p1_failed + 1 ))
  state "phase 1" "arm ${names[$i]} rc=$rc"
done
log "phase 1 done — $p1_failed arm(s) failed"
bash "$HERE/make_plots.sh" 2>&1 | tee -a "$LOG"
state "phase 1 done" "$p1_failed arm(s) failed"

# ---- Phase 2 -----------------------------------------------------------------
# The picker refuses an incomplete phase 1, which is the card's own rule, so a
# failed arm stops the study here rather than choosing two arms out of a table
# with a hole in it.
if [ -n "${ARMS:-}" ]; then
  arms="$ARMS"
else
  arms="$(python3 "$HERE/pick_phase2_arms.py" --scores "$CF401_RESULTS/scores.csv")"
  rc=$?
  [ $rc -eq 0 ] || { log "ABORT: the picker exited rc=$rc"
                     state "phase 2" "the picker refused phase 1"; exit $rc; }
fi
log "phase 2 arms: $arms"
state "phase 2" "arms $arms"

# One `phase2.sh` per arm, each with its own head lock. `head_vram_gate` takes
# `$GPU_GATE_LOCKDIR/cf373_head_gpu<N>.lock`, so a lock directory per arm lets
# one head per arm train at a time. Its VRAM check still runs against the real
# card, so the second arm waits when the first head leaves too little.
p2_pids=(); p2_names=()
for k in $arms; do
  lockdir="/tmp/cf401_head_k${k}"
  mkdir -p "$lockdir"
  log "phase 2 arm k=$k START (lockdir=$lockdir)"
  ARMS="$k" BB_GPU="$BB_GPU" HEAD_BG=0 GPU_GATE_LOCKDIR="$lockdir" \
    nohup bash "$HERE/phase2.sh" >>"$CF401_RESULTS/phase2_k${k}.out" 2>&1 &
  p2_pids+=($!); p2_names+=("k=$k")
  sleep "$STAGGER"
done

p2_failed=0
for i in "${!p2_pids[@]}"; do
  wait "${p2_pids[$i]}"; rc=$?
  log "phase 2 arm ${p2_names[$i]} rc=$rc"
  [ $rc -eq 0 ] || p2_failed=$(( p2_failed + 1 ))
  state "phase 2" "arm ${p2_names[$i]} rc=$rc"
done

bash "$HERE/make_plots.sh" 2>&1 | tee -a "$LOG"
log "STUDY DONE — $p1_failed phase-1 arm(s) and $p2_failed phase-2 arm(s) failed"
state "done" "$p1_failed phase-1 arm(s) and $p2_failed phase-2 arm(s) failed"
[ $(( p1_failed + p2_failed )) -eq 0 ] || exit 1
